"""
Cliente Admin para gerenciamento de usuários do sistema de reconhecimento facial.
"""
import requests
import cv2
import getpass
import os

# Configuração do servidor - ALTERE PARA O IP DO SEU RASPBERRY PI
BASE_URL = os.environ.get("SERVER_URL", "http://localhost:5000")
AUTH_TOKEN = None


def login():
    """Solicita login e armazena o token."""
    global AUTH_TOKEN
    print("\n" + "="*50)
    print("  LOGIN DE ADMINISTRADOR")
    print("="*50)
    print(f"Servidor: {BASE_URL}")
    print()
    
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    
    try:
        url = f"{BASE_URL}/admin/login"
        response = requests.post(url, json={"username": username, "password": password}, timeout=10)
        
        if response.status_code == 200:
            AUTH_TOKEN = response.json().get('access_token')
            print("\n✓ Login realizado com sucesso!")
            return True
        else:
            error = response.json().get('error', 'Erro desconhecido')
            print(f"\n✗ Erro no login: {error}")
            return False
    except requests.ConnectionError:
        print(f"\n✗ Erro: Não foi possível conectar ao servidor em {BASE_URL}")
        print("  Verifique se o servidor está rodando e o IP está correto.")
        return False
    except requests.Timeout:
        print("\n✗ Erro: Timeout na conexão.")
        return False


def get_auth_headers():
    """Helper para criar os headers de autenticação."""
    if not AUTH_TOKEN:
        raise Exception("Não autenticado. Faça login primeiro.")
    return {'Authorization': f'Bearer {AUTH_TOKEN}'}

def capture_image_from_cam():
    """Abre a webcam e captura um frame ao pressionar ESPAÇO."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Erro: Não foi possível abrir a webcam.")
        return None
    
    # Configura resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    print("\n📷 Webcam aberta.")
    print("   Pressione ESPAÇO para capturar a foto")
    print("   Pressione ESC para cancelar")
    
    frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Erro ao ler frame da câmera.")
            break
        
        # Adiciona instruções na imagem
        display = frame.copy()
        cv2.putText(display, "ESPACO: Capturar | ESC: Cancelar", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capturar Foto - Admin', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # ESPAÇO
            print("✓ Foto capturada!")
            break
        elif key == 27:  # ESC
            print("✗ Captura cancelada.")
            frame = None
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return frame

# --- Funções do Menu ---

def add_user():
    """Fluxo de adicionar novo usuário."""
    print("\n" + "-"*50)
    print("  ADICIONAR NOVO USUÁRIO")
    print("-"*50)
    
    name = input("Digite o nome do novo usuário: ").strip()
    if not name:
        print("✗ Nome não pode ser vazio.")
        return
        
    image_frame = capture_image_from_cam()
    
    if image_frame is None:
        print("✗ Captura cancelada ou falhou.")
        return
        
    print("Enviando para o servidor...")
    
    ret, img_encoded = cv2.imencode('.jpg', image_frame)
    if not ret:
        print("✗ Erro ao codificar imagem.")
        return

    try:
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users/add"
        
        files = {'image': ('user.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'name': name}
        
        response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        
        if response.status_code == 201:
            result = response.json()
            print(f"\n✓ Sucesso! Usuário '{name}' adicionado com ID: {result['user_id']}")
        else:
            error = response.json().get('error', 'Erro desconhecido')
            print(f"\n✗ Erro do servidor: {error}")
            
    except requests.Timeout:
        print("✗ Timeout: O servidor demorou muito para responder.")
    except Exception as e:
        print(f"✗ Erro na requisição: {e}")


def list_users():
    """Lista todos os usuários cadastrados."""
    print("\n" + "-"*50)
    print("  USUÁRIOS CADASTRADOS")
    print("-"*50)
    
    try:
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            users = response.json()
            if not users:
                print("Nenhum usuário cadastrado.")
            else:
                print(f"{'ID':<6} | {'Nome':<30}")
                print("-"*40)
                for user in users:
                    print(f"{user['id']:<6} | {user['name']:<30}")
                print(f"\nTotal: {len(users)} usuário(s)")
        else:
            error = response.json().get('error', 'Erro desconhecido')
            print(f"✗ Erro: {error}")
            
    except Exception as e:
        print(f"✗ Erro na requisição: {e}")


def delete_user():
    """Remove um usuário do sistema."""
    print("\n" + "-"*50)
    print("  DELETAR USUÁRIO")
    print("-"*50)
    
    try:
        user_id = input("Digite o ID do usuário a ser deletado: ").strip()
        if not user_id.isdigit():
            print("✗ ID inválido. Deve ser um número.")
            return
        
        confirm = input(f"Confirma exclusão do usuário ID {user_id}? (s/N): ").strip().lower()
        if confirm != 's':
            print("Operação cancelada.")
            return
            
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users/{user_id}"
        
        response = requests.delete(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print(f"\n✓ Usuário ID {user_id} deletado com sucesso!")
        elif response.status_code == 404:
            print(f"✗ Usuário ID {user_id} não encontrado.")
        else:
            error = response.json().get('error', 'Erro desconhecido')
            print(f"✗ Erro: {error}")

    except Exception as e:
        print(f"✗ Erro na requisição: {e}")


def check_server_health():
    """Verifica o status do servidor."""
    print("\n" + "-"*50)
    print("  STATUS DO SERVIDOR")
    print("-"*50)
    
    try:
        url = f"{BASE_URL}/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data.get('status', 'desconhecido')}")
            print(f"Modelo carregado: {'Sim' if data.get('model_loaded') else 'Não'}")
            print(f"Usuários no modelo: {data.get('users_in_model', 0)}")
        else:
            print(f"✗ Servidor retornou código: {response.status_code}")
            
    except requests.ConnectionError:
        print(f"✗ Não foi possível conectar ao servidor em {BASE_URL}")
    except Exception as e:
        print(f"✗ Erro: {e}")


def main():
    """Loop principal do menu."""
    print("\n" + "="*50)
    print("  SISTEMA DE RECONHECIMENTO FACIAL")
    print("  Cliente Administrativo")
    print("="*50)
    
    if not login():
        return
        
    while True:
        print("\n" + "="*50)
        print("  MENU PRINCIPAL")
        print("="*50)
        print("1. Adicionar Usuário")
        print("2. Listar Usuários")
        print("3. Deletar Usuário")
        print("4. Status do Servidor")
        print("5. Sair")
        print("-"*50)
        
        choice = input("Escolha uma opção (1-5): ").strip()
        
        if choice == '1':
            add_user()
        elif choice == '2':
            list_users()
        elif choice == '3':
            delete_user()
        elif choice == '4':
            check_server_health()
        elif choice == '5':
            print("\nSaindo...")
            break
        else:
            print("✗ Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
