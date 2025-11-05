# servidor_pi/auth_utils.py
import bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

def hash_password(password):
    """Gera um hash seguro para a senha usando bcrypt."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hash_bytes = bcrypt.hashpw(password_bytes, salt)
    # Retorna o hash como string para armazenar no DB
    return hash_bytes.decode('utf-8')

def check_password(password, hashed_password):
    """Verifica se a senha fornecida corresponde ao hash."""
    if not hashed_password:
        return False
    password_bytes = password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

# Funções de token (são importadas diretamente do flask_jwt_extended,
# mas são listadas aqui para clareza conceitual)
#
# create_access_token(identity)
#   - Usado no endpoint de login para criar um novo token.
#
# @jwt_required()
#   - Um decorador usado nos endpoints protegidos.
#
# get_jwt_identity()
#   - (Opcional) Usado dentro de um endpoint protegido para
#     descobrir quem é o usuário logado.