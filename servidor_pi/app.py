"""
Servidor Flask para Sistema de Reconhecimento Facial.
Otimizado para Raspberry Pi 3 usando OpenCV LBPH.

Features:
- ThreadPool para múltiplas conexões simultâneas
- Interface Web administrativa
- Dashboard de estatísticas
- Backup automático de dados
- Detecção de tentativas de invasão
"""
import os
import cv2
import numpy as np
import click
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from flask import (
    Flask, jsonify, request, render_template, 
    redirect, url_for, session, flash
)
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_bcrypt import Bcrypt

import servidor_pi.database as db
import servidor_pi.auth_utils as auth
import servidor_pi.face_utils as face
import servidor_pi.security as security
import servidor_pi.backup as backup

app = Flask(__name__)

# Configurações
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "chave-secreta-de-desenvolvimento-mude-depois")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "session-secret-key-mude-em-producao")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
app.config["TEMPLATES_AUTO_RELOAD"] = True

jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# ThreadPool para processamento paralelo
# Otimizado para Raspberry Pi 3 (recursos limitados)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def get_client_ip():
    """Obtém o IP real do cliente, considerando proxies."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'


# ============================================
# DECORATORS
# ============================================

def web_login_required(f):
    """Decorator para proteger rotas web."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_user' not in session:
            return redirect(url_for('web_login'))
        return f(*args, **kwargs)
    return decorated_function


def check_blocked_ip(f):
    """Decorator para verificar se IP está bloqueado."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = get_client_ip()
        if security.is_ip_blocked(ip):
            remaining = security.get_block_remaining_time(ip)
            return jsonify({
                "error": f"IP bloqueado. Tente novamente em {remaining // 60} minutos.",
                "blocked": True,
                "remaining_seconds": remaining
            }), 403
        return f(*args, **kwargs)
    return decorated_function


# ============================================
# COMANDOS CLI
# ============================================

@app.cli.command("init-db")
def init_db_command():
    """Inicializa o banco de dados."""
    db.init_db()
    click.echo("Banco de dados inicializado.")


@app.cli.command("add-admin")
@click.argument("username")
@click.argument("password")
def add_admin_command(username, password):
    """Adiciona um administrador: flask add-admin <username> <password>"""
    hash_pw = auth.hash_password(password)
    db.add_admin(username, hash_pw)
    click.echo(f"Admin '{username}' adicionado com sucesso.")


@app.cli.command("clear-model")
def clear_model_command():
    """Limpa o modelo LBPH treinado."""
    face.clear_model()
    click.echo("Modelo LBPH removido.")


@app.cli.command("model-stats")
def model_stats_command():
    """Mostra estatísticas do modelo LBPH."""
    stats = face.get_model_stats()
    click.echo(f"Usuários no modelo: {stats['total_users']}")
    click.echo(f"Modelo existe: {stats['model_exists']}")
    click.echo(f"Limiar: {stats['threshold']}")
    for user in stats['users']:
        click.echo(f"  - {user['name']} (ID: {user['user_id']})")


@app.cli.command("backup")
def backup_command():
    """Cria um backup manual."""
    result = backup.create_backup(manual=True)
    if result['success']:
        click.echo(f"Backup criado: {result['backup_name']}")
    else:
        click.echo(f"Erro: {result.get('error')}")


@app.cli.command("start-auto-backup")
def start_auto_backup_command():
    """Inicia backup automático."""
    backup.start_auto_backup()
    click.echo("Backup automático iniciado.")


# ============================================
# ENDPOINTS DA API REST
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde do servidor."""
    stats = face.get_model_stats()
    return jsonify({
        "status": "ok",
        "model_loaded": stats['model_exists'],
        "users_in_model": stats['total_users'],
        "threadpool_workers": MAX_WORKERS
    })


@app.route('/admin/login', methods=['POST'])
@check_blocked_ip
def admin_login():
    """Autentica um admin e retorna um token JWT."""
    ip = get_client_ip()
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "JSON inválido"}), 400
    
    username = data.get("username", None)
    password = data.get("password", None)
    
    if not username or not password:
        return jsonify({"error": "Username e password são obrigatórios"}), 400
    
    admin_hash = db.get_admin_hash(username)
    
    if admin_hash and auth.check_password(password, admin_hash):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    
    # Registra tentativa falha
    security.record_failed_attempt(ip, f"Login falho para '{username}'")
    return jsonify({"error": "Credenciais inválidas"}), 401


@app.route('/admin/users/add', methods=['POST'])
@jwt_required()
def add_user():
    """Adiciona um novo usuário com sua face ao modelo LBPH."""
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltando 'image' ou 'name' no formulário"}), 400
    
    file = request.files['image']
    name = request.form['name'].strip()
    
    if not name:
        return jsonify({"error": "Nome não pode ser vazio"}), 400
    
    try:
        # Processa em thread separada para não bloquear
        def process_face():
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if image_np is None:
                raise ValueError("Imagem inválida")
            
            user_id = db.add_user(name)
            face.add_face_to_model(user_id, name, image_np)
            return user_id
        
        future = executor.submit(process_face)
        user_id = future.result(timeout=30)
        
        return jsonify({
            "success": True, 
            "user_id": user_id, 
            "name": name,
            "message": f"Usuário '{name}' cadastrado com sucesso"
        }), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


@app.route('/admin/users', methods=['GET'])
@jwt_required()
def list_users():
    """Lista todos os usuários cadastrados."""
    users_list = db.get_all_users()
    return jsonify(users_list)


@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """Deleta um usuário e remove do modelo LBPH."""
    try:
        if not db.user_exists(user_id):
            return jsonify({"error": "Usuário não encontrado"}), 404
        
        # Remove do modelo LBPH
        face.remove_user_from_model(user_id)
        
        # Remove do banco
        db.delete_user_by_id(user_id)
        
        return jsonify({
            "success": True, 
            "deleted_id": user_id,
            "message": "Usuário removido com sucesso"
        })
    except Exception as e:
        return jsonify({"error": f"Erro ao deletar: {str(e)}"}), 500


@app.route('/admin/model/stats', methods=['GET'])
@jwt_required()
def model_stats():
    """Retorna estatísticas do modelo LBPH."""
    stats = face.get_model_stats()
    return jsonify(stats)


@app.route('/admin/security/stats', methods=['GET'])
@jwt_required()
def security_stats():
    """Retorna estatísticas de segurança."""
    return jsonify({
        "stats": security.get_access_statistics(24),
        "blocked_ips": security.get_blocked_ips(),
        "alerts": security.get_recent_alerts(50)
    })


@app.route('/admin/backup/create', methods=['POST'])
@jwt_required()
def api_create_backup():
    """Cria um backup manual via API."""
    result = backup.create_backup(manual=True)
    return jsonify(result)


@app.route('/admin/backup/list', methods=['GET'])
@jwt_required()
def api_list_backups():
    """Lista backups disponíveis."""
    return jsonify({
        "backups": backup.list_backups(),
        "status": backup.get_backup_status()
    })


@app.route('/verify', methods=['POST'])
@check_blocked_ip
def verify_access():
    """Endpoint público para verificar uma face e liberar acesso."""
    ip = get_client_ip()
    
    if 'image' not in request.files:
        return jsonify({
            "acesso_liberado": False, 
            "nome": "Desconhecido",
            "error": "Faltando 'image' no formulário"
        }), 400
        
    file = request.files['image']
    
    try:
        def process_verification():
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if image_np is None:
                raise ValueError("Imagem inválida")
            
            return face.recognize_face(image_np)
        
        # Processa em thread separada
        future = executor.submit(process_verification)
        match_result = future.result(timeout=30)
        
        if match_result:
            # Acesso liberado
            security.record_successful_access(ip, match_result['name'])
            return jsonify({
                "acesso_liberado": True, 
                "nome": match_result['name'],
                "user_id": match_result['user_id'],
                "confianca": round(match_result['confidence'], 2)
            })
        else:
            # Acesso negado
            was_blocked = security.record_failed_attempt(ip, "Face não reconhecida")
            response = {
                "acesso_liberado": False, 
                "nome": "Desconhecido"
            }
            if was_blocked:
                response["warning"] = "IP bloqueado por tentativas excessivas"
            return jsonify(response)
            
    except ValueError as e:
        security.record_failed_attempt(ip, str(e))
        return jsonify({
            "acesso_liberado": False, 
            "nome": "Desconhecido", 
            "error": str(e)
        })
    except Exception as e:
        return jsonify({
            "acesso_liberado": False,
            "nome": "Desconhecido",
            "error": f"Erro interno: {str(e)}"
        }), 500


# ============================================
# INTERFACE WEB
# ============================================

@app.route('/web/login', methods=['GET', 'POST'])
def web_login():
    """Página de login da interface web."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin_hash = db.get_admin_hash(username)
        
        if admin_hash and auth.check_password(password, admin_hash):
            session['admin_user'] = username
            return redirect(url_for('web_dashboard'))
        
        return render_template('login.html', error="Credenciais inválidas")
    
    return render_template('login.html')


@app.route('/web/logout')
def web_logout():
    """Logout da interface web."""
    session.pop('admin_user', None)
    return redirect(url_for('web_login'))


@app.route('/web/')
@app.route('/web/dashboard')
@web_login_required
def web_dashboard():
    """Dashboard principal."""
    stats = security.get_access_statistics(24)
    model_stats_data = face.get_model_stats()
    blocked_ips = security.get_blocked_ips()
    alerts = security.get_recent_alerts(10)
    backup_status = backup.get_backup_status()
    
    return render_template('dashboard.html',
        stats=stats,
        model_stats=model_stats_data,
        blocked_ips=blocked_ips,
        alerts=alerts,
        backup_status=backup_status
    )


@app.route('/web/users')
@web_login_required
def web_users():
    """Gerenciamento de usuários."""
    users = db.get_all_users()
    model_stats_data = face.get_model_stats()
    
    # IDs dos usuários no modelo
    model_user_ids = [u['user_id'] for u in model_stats_data.get('users', [])]
    
    return render_template('users.html',
        users=users,
        model_stats=model_stats_data,
        model_user_ids=model_user_ids
    )


@app.route('/web/users/add', methods=['POST'])
@web_login_required
def web_add_user():
    """Adiciona usuário via interface web."""
    if 'image' not in request.files or 'name' not in request.form:
        flash('Faltando nome ou imagem', 'danger')
        return redirect(url_for('web_users'))
    
    file = request.files['image']
    name = request.form['name'].strip()
    
    if not name:
        flash('Nome não pode ser vazio', 'danger')
        return redirect(url_for('web_users'))
    
    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image_np is None:
            flash('Imagem inválida', 'danger')
            return redirect(url_for('web_users'))
        
        user_id = db.add_user(name)
        face.add_face_to_model(user_id, name, image_np)
        
        flash(f'Usuário "{name}" cadastrado com sucesso!', 'success')
    except ValueError as e:
        flash(f'Erro: {str(e)}', 'danger')
    except Exception as e:
        flash(f'Erro interno: {str(e)}', 'danger')
    
    return redirect(url_for('web_users'))


@app.route('/web/users/<int:user_id>/delete', methods=['POST'])
@web_login_required
def web_delete_user(user_id):
    """Remove usuário via interface web."""
    try:
        if not db.user_exists(user_id):
            return jsonify({"success": False, "error": "Usuário não encontrado"})
        
        face.remove_user_from_model(user_id)
        db.delete_user_by_id(user_id)
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/web/security')
@web_login_required
def web_security():
    """Página de segurança."""
    stats = security.get_access_statistics(24)
    blocked_ips = security.get_blocked_ips()
    alerts = security.get_recent_alerts(50)
    
    config = {
        "max_attempts": security.MAX_FAILED_ATTEMPTS,
        "block_duration": security.BLOCK_DURATION_MINUTES
    }
    
    return render_template('security.html',
        stats=stats,
        blocked_ips=blocked_ips,
        alerts=alerts,
        config=config
    )


@app.route('/web/security/unblock', methods=['POST'])
@web_login_required
def web_unblock_ip():
    """Desbloqueia um IP."""
    ip = request.form.get('ip')
    if ip and security.unblock_ip(ip):
        flash(f'IP {ip} desbloqueado com sucesso', 'success')
    else:
        flash('Erro ao desbloquear IP', 'danger')
    return redirect(url_for('web_security'))


@app.route('/web/backups')
@web_login_required
def web_backups():
    """Página de backups."""
    backups_list = backup.list_backups()
    status = backup.get_backup_status()
    
    return render_template('backups.html',
        backups=backups_list,
        status=status
    )


@app.route('/web/backups/create', methods=['POST'])
@web_login_required
def web_create_backup():
    """Cria backup manual."""
    result = backup.create_backup(manual=True)
    if result['success']:
        flash(f'Backup criado: {result["backup_name"]}', 'success')
    else:
        flash(f'Erro: {result.get("error")}', 'danger')
    return redirect(url_for('web_backups'))


@app.route('/web/backups/restore', methods=['POST'])
@web_login_required
def web_restore_backup():
    """Restaura um backup."""
    backup_name = request.form.get('backup_name')
    result = backup.restore_backup(backup_name)
    if result['success']:
        flash(f'Backup restaurado: {backup_name}', 'success')
        # Recarrega modelo LBPH
        face._recognizer = None
        face._label_map = {}
    else:
        flash(f'Erro: {result.get("error")}', 'danger')
    return redirect(url_for('web_backups'))


@app.route('/web/backups/delete', methods=['POST'])
@web_login_required
def web_delete_backup():
    """Exclui um backup."""
    backup_name = request.form.get('backup_name')
    result = backup.delete_backup(backup_name)
    if result['success']:
        flash(f'Backup excluído: {backup_name}', 'success')
    else:
        flash(f'Erro: {result.get("error")}', 'danger')
    return redirect(url_for('web_backups'))


@app.route('/web/backups/toggle-auto', methods=['POST'])
@web_login_required
def web_toggle_auto_backup():
    """Ativa/desativa backup automático."""
    action = request.form.get('action')
    if action == 'start':
        backup.start_auto_backup()
        flash('Backup automático ativado', 'success')
    else:
        backup.stop_auto_backup()
        flash('Backup automático desativado', 'warning')
    return redirect(url_for('web_backups'))


@app.route('/web/logs')
@web_login_required
def web_logs():
    """Página de logs."""
    limit = request.args.get('limit', 100, type=int)
    logs = security.get_access_log(limit)
    
    return render_template('logs.html', logs=logs, limit=limit)


# ============================================
# INICIALIZAÇÃO
# ============================================

def create_app():
    """Factory function para criar a aplicação."""
    return app


if __name__ == '__main__':
    # Inicia backup automático
    backup.start_auto_backup()
    
    # Modo desenvolvimento - não usar em produção
    print(f"[SERVER] ThreadPool iniciado com {MAX_WORKERS} workers")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
