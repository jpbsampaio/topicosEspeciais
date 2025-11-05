# servidor_pi/database.py
import sqlite3
import numpy as np

DATABASE_FILE = 'servidor_pi/database.db'

def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa o esquema do banco de dados."""
    # CORREÇÃO: A variável estava vazia. Definido o esquema SQL.
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
            name TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """
    ]
    
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cursor = conn.cursor()
            for statement in sql_statements:
                cursor.execute(statement)
            conn.commit()
            print("Banco de dados inicializado com sucesso.")
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

def add_face_encoding(user_id, encoding_vector):
    """Adiciona um encoding facial (vetor numpy) para um usuário."""
    # Converte o vetor numpy para bytes (BLOB)
    # CORREÇÃO: Padronizando para float32
    encoding_blob = encoding_vector.astype(np.float32).tobytes()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO face_encodings (user_id, encoding) VALUES (?,?)",
            (user_id, encoding_blob)
        )
        conn.commit()

def get_all_users_with_names():
    """Lista todos os usuários (ID e Nome)."""
    with get_db_connection() as conn:
        users = conn.execute("SELECT id, name FROM users").fetchall()
        return [dict(user) for user in users]

def delete_user_by_id(user_id):
    """Deleta um usuário pelo ID. O 'ON DELETE CASCADE' cuidará dos encodings."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id =?", (user_id,))
        conn.commit()

def get_all_encodings():
    """
    Busca todos os encodings faciais e os junta com os nomes dos usuários.
    Retorna uma lista de tuplas: (user_id, name, encoding_vector)
    """
    with get_db_connection() as conn:
        # CORREÇÃO: O JOIN estava errado, "u.user_id" não existe, o correto é "u.id"
        rows = conn.execute("""
            SELECT u.id, u.name, fe.encoding
            FROM face_encodings fe
            JOIN users u ON u.id = fe.user_id
        """).fetchall()

        # CORREÇÃO: A lista estava sendo inicializada vazia (SyntaxError)
        encodings_list = []
        for row in rows:
            # Converte o BLOB de volta para um vetor numpy
            # CORREÇÃO: Garantindo que o dtype seja float32 (para bater com o save)
            encoding_vector = np.frombuffer(row['encoding'], dtype=np.float32)
            encodings_list.append({
                "user_id": row['id'],
                "name": row['name'],
                "encoding": encoding_vector
            })
        return encodings_list

if __name__ == '__main__':
    """Permite a inicialização via 'python -m servidor_pi.database'"""
    print("Inicializando o banco de dados...")
    init_db()