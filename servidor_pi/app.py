# servidor_pi/app.py
import numpy as np
import cv2
import click # CORREÇÃO: Importado para os comandos CLI
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_bcrypt import Bcrypt

# Importa nossos modulos locais
import servidor_pi.database as db
import servidor_pi.auth_utils as auth
import servidor_pi.face_utils as face

app = Flask(__name__)

# --- Configuração ---
# Mude isso para uma string aleatoria e segura em producao!
# CORREÇÃO: A configuração estava sintaticamente incorreta.
app.config["JWT_SECRET_KEY"] = "chave-secreta-de-desenvolvimento-mude-depois"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # Limite de upload de 16MB

# Inicializa extensoes
jwt = JWTManager(app)
bcrypt = Bcrypt(app)

# --- Comandos CLI ---
@app.cli.command("init-db")
def init_db_command():
    """Comando Flask CLI para inicializar o banco de dados."""
    db.init_db()

@app.cli.command("add-admin")
@click.argument("username") # CORREÇÃO: Adicionado argumento click
@click.argument("password") # CORREÇÃO: Adicionado argumento click
def add_admin_command(username, password):
    """(Helper) Comando para adicionar um admin manualmente."""
    hash_pw = auth.hash_password(password)
    db.add_admin(username, hash_pw)
    print(f"Admin {username} adicionado com sucesso.")

# --- Endpoints da API ---

@app.route('/admin/login', methods=['POST']) # CORREÇÃO: methods=['POST']
def admin_login():
    """RF-S03: Autentica um admin e retorna um token JWT."""
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    
    admin_hash = db.get_admin_hash(username)
    
    if admin_hash and auth.check_password(password, admin_hash):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    
    return jsonify({"error": "Credenciais invalidas"}), 401


@app.route('/admin/users/add', methods=['POST']) # CORREÇÃO: methods=['POST']
@jwt_required() # RNF-01: Protegido por JWT
def add_user():
    """RF-S06, RF-S07: Adiciona um novo usuario e seu encoding facial."""
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltando 'image' ou 'name' no formulario"}), 400
    
    file = request.files['image']
    name = request.form['name']
    
    try:
        # Processa a imagem em memoria (nao salva no disco)
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Gera o embedding facial (pode levantar ValueError - RNF-05)
        embedding = face.generate_embedding(image_np)
        
        # Salva no banco de dados
        user_id = db.add_user(name)
        db.add_face_encoding(user_id, embedding)
        
        return jsonify({
            "success": True, 
            "user_id": user_id, 
            "name": name
        }), 201

    except ValueError as e:
        # Captura erros do face_utils (ex: "Nenhum rosto detectado")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


@app.route('/admin/users', methods=['GET']) # CORREÇÃO: methods=['GET']
@jwt_required() # RNF-01: Protegido por JWT
def list_users():
    """RF-S08: Lista todos os usuarios cadastrados."""
    users_list = db.get_all_users_with_names()
    return jsonify(users_list)


@app.route('/admin/users/<int:user_id>', methods=['DELETE']) # CORREÇÃO: methods=['DELETE']
@jwt_required() # RNF-01: Protegido por JWT
def delete_user(user_id):
    """RF-S09: Deleta um usuario (e seus encodings via CASCADE)."""
    try:
        db.delete_user_by_id(user_id)
        return jsonify({"success": True, "deleted_id": user_id})
    except Exception as e:
        return jsonify({"error": f"Erro ao deletar: {str(e)}"}), 500


@app.route('/verify', methods=['POST']) # CORREÇÃO: methods=['POST']
def verify_access():
    """RF-S04, RF-S05: Endpoint publico para verificar uma face."""
    if 'image' not in request.files:
        return jsonify({"error": "Faltando 'image' no formulario"}), 400
        
    file = request.files['image']
    
    try:
        # 1. Processa a imagem recebida
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 2. Gera o embedding da imagem (pode falhar RNF-05)
        new_embedding = face.generate_embedding(image_np)

        # 3. Busca todos os embeddings do DB
        all_encodings = db.get_all_encodings()
        if not all_encodings:
             return jsonify({"acesso_liberado": False, "nome": "Desconhecido"})

        # 4. Encontra a correspondencia (Logica 1:N)
        match_result = face.find_match(new_embedding, all_encodings)
        
        # 5. Retorna o resultado
        if match_result:
            return jsonify({
                "acesso_liberado": True, 
                "nome": match_result['name']
            })
        else:
            return jsonify({
                "acesso_liberado": False, 
                "nome": "Desconhecido"
            })
            
    except ValueError as e:
        # (RNF-05) Se 'generate_embedding' falhar (ex: sem rosto)
        return jsonify({"acesso_liberado": False, "nome": "Desconhecido", "error": str(e)})
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    # NUNCA use app.run() em producao. Use Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)