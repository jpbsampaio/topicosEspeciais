# servidor_pi/database.py
"""
Módulo de banco de dados SQLite para o sistema de reconhecimento facial.
Armazena admins e usuários. O modelo LBPH é gerenciado por face_utils.py.
"""
import sqlite3
import os

DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'database.db')


def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inicializa o esquema do banco de dados."""
    sql_statements = [
        """
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for statement in sql_statements:
                cursor.execute(statement)
            conn.commit()
            print(f"Banco de dados inicializado em: {DATABASE_FILE}")
    except sqlite3.OperationalError as e:
        print(f"Erro ao inicializar o banco de dados: {e}")


def add_admin(username, password_hash):
    """Adiciona um novo administrador ao banco de dados."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?,?)",
            (username, password_hash)
        )
        conn.commit()


def get_admin_hash(username):
    """Busca o hash da senha de um admin pelo username."""
    with get_db_connection() as conn:
        admin = conn.execute(
            "SELECT password_hash FROM admins WHERE username =?", (username,)
        ).fetchone()
        if admin:
            return admin['password_hash']
    return None


def add_user(name):
    """Adiciona um novo usuário e retorna seu ID."""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name) VALUES (?)", (name,)
        )
        conn.commit()
        return cursor.lastrowid


def get_user_by_id(user_id):
    """Busca um usuário pelo ID."""
    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, name FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user:
            return dict(user)
    return None


def get_all_users():
    """Lista todos os usuários (ID e Nome)."""
    with get_db_connection() as conn:
        users = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [dict(user) for user in users]


def delete_user_by_id(user_id):
    """Deleta um usuário pelo ID."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()


def user_exists(user_id):
    """Verifica se um usuário existe."""
    with get_db_connection() as conn:
        result = conn.execute(
            "SELECT 1 FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return result is not None


if __name__ == '__main__':
    """Permite a inicialização via 'python -m servidor_pi.database'"""
    print("Inicializando o banco de dados...")
    init_db()
