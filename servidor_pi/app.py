import numpy as np
import cv2
import click
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_bcrypt import Bcrypt

import servidor_pi.database as db
import servidor_pi.auth_utils as auth
import servidor_pi.face_utils as face

app = Flask(__name__)

app.config["JWT_SECRET_KEY"] = "chave-secreta-de-desenvolvimento-mude-depois"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

jwt = JWTManager(app)
bcrypt = Bcrypt(app)

@app.cli.command("init-db")
def init_db_command():
    """Comando Flask CLI para inicializar o banco de dados."""
    db.init_db()

@app.cli.command("add-admin")
@click.argument("username")
@click.argument("password")
def add_admin_command(username, password):
    """(Helper) Comando para adicionar um admin manualmente."""
    hash_pw = auth.hash_password(password)
    db.add_admin(username, hash_pw)
    print(f"Admin {username} adicionado com sucesso.")

# --- Endpoints da API ---

@app.route('/admin/login', methods=['POST'])
def admin_login():
    """RF-S03: Autentica um admin e retorna um token JWT."""
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    
    admin_hash = db.get_admin_hash(username)
    
    if admin_hash and auth.check_password(password, admin_hash):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    
    return jsonify({"error": "Credenciais invalidas"}), 401


@app.route('/admin/users/add', methods=['POST'])
@jwt_required()
def add_user():
    """RF-S06, RF-S07: Adiciona um novo usuario e seu encoding facial."""
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Faltando 'image' ou 'name' no formulario"}), 400
    
    file = request.files['image']
    name = request.form['name']
    
    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        embedding = face.generate_embedding(image_np)
        
        user_id = db.add_user(name)
        db.add_face_encoding(user_id, embedding)
        
        return jsonify({
            "success": True, 
            "user_id": user_id, 
            "name": name
        }), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500


@app.route('/admin/users', methods=['GET'])
@jwt_required()
def list_users():
    """RF-S08: Lista todos os usuarios cadastrados."""
    users_list = db.get_all_users_with_names()
    return jsonify(users_list)


@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """RF-S09: Deleta um usuario (e seus encodings via CASCADE)."""
    try:
        db.delete_user_by_id(user_id)
        return jsonify({"success": True, "deleted_id": user_id})
    except Exception as e:
        return jsonify({"error": f"Erro ao deletar: {str(e)}"}), 500


@app.route('/verify', methods=['POST'])
def verify_access():
    """RF-S04, RF-S05: Endpoint publico para verificar uma face."""
    if 'image' not in request.files:
        return jsonify({"error": "Faltando 'image' no formulario"}), 400
        
    file = request.files['image']
    
    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        new_embedding = face.generate_embedding(image_np)

        all_encodings = db.get_all_encodings()
        if not all_encodings:
             return jsonify({"acesso_liberado": False, "nome": "Desconhecido"})

        match_result = face.find_match(new_embedding, all_encodings)
        
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
        return jsonify({"acesso_liberado": False, "nome": "Desconhecido", "error": str(e)})
    except Exception as e:
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
