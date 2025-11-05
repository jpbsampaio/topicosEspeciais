# cliente_admin/admin.py
import requests
import cv2
import getpass # Para esconder a senha

# --- Configuracao ---
# CORREÇÃO: 127.0.0.1 é localhost. Para funcionar em outra máquina,
# você DEVE colocar o IP real do seu Raspberry Pi na rede.
BASE_URL = "http://<IP_DO_SEU_RASPBERRY_PI>:5000" 
AUTH_TOKEN = None # RF-A03: Armazena o token em memoria

# --- Funcoes de Helper da API ---

def login():
    """RF-A01, RF-A02: Solicita login e armazena o token."""
    global AUTH_TOKEN
    print("--- Login de Administrador ---")
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    
    try:
        url = f"{BASE_URL}/admin/login"
        response = requests.post(url, json={"username": username, "password": password})
        
        if response.status_code == 200:
            AUTH_TOKEN = response.json().get('access_token')
            print("Login realizado com sucesso!\n")
            return True
        else:
            print(f"Erro no login: {response.json().get('error', 'Erro')}\n")
            return False
    except requests.ConnectionError:
        print(f"Erro: Nao foi possivel conectar ao servidor em {url}.")
        return False

def get_auth_headers():
    """Helper para criar os headers de autenticacao."""
    if not AUTH_TOKEN:
        raise Exception("Nao autenticado. Faca login primeiro.")
    return {'Authorization': f'Bearer {AUTH_TOKEN}'}

def capture_image_from_cam():
    """Abre a webcam e captura um frame ao pressionar ESPACO."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Nao foi possivel abrir a webcam.")
        return None
        
    print("Pressione ESPACO para capturar a foto, ESC para cancelar.")
    frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Exibe instrucoes no video
        cv2.putText(frame, "Pressione ESPACO para capturar", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Capturar Foto Admin', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # 32 eh o ASCII para ESPACO
            break
        elif key == 27: # 27 eh o ASCII para ESC
            frame = None
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return frame

# --- Funcoes do Menu (RF-A04) ---

def add_user():
    """RF-A05: Fluxo de adicionar novo usuario."""
    print("\n--- 1. Adicionar Novo Usuario ---")
    name = input("Digite o nome do novo usuario: ")
    if not name:
        print("Nome nao pode ser vazio.")
        return
        
    image_frame = capture_image_from_cam()
    
    if image_frame is None:
        print("Captura cancelada ou falhou.")
        return
        
    print("Foto capturada. Enviando para o servidor...")
    
    # Codifica a imagem para JPEG em memoria
    ret, img_encoded = cv2.imencode('.jpg', image_frame)
    if not ret:
        print("Erro ao codificar imagem.")
        return

    try:
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users/add"
        
        # Prepara o payload multipart/form-data
        files = {'image': ('user.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'name': name}
        
        response = requests.post(url, headers=headers, data=data, files=files)
        
        if response.status_code == 201:
            print(f"Sucesso! Usuario '{name}' adicionado com ID: {response.json()['user_id']}")
        else:
            # RNF-04 é tratado aqui (RNF-05 no server)
            print(f"Erro do servidor: {response.json().get('error', 'Erro')}") 
            
    except Exception as e:
        print(f"Erro na requisicao: {e}")


def list_users():
    """RF-A06: Fluxo de listar usuarios."""
    print("\n--- 2. Listar Usuarios ---")
    try:
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            users = response.json()
            if not users:
                print("Nenhum usuario cadastrado.")
            else:
                print("ID | Nome")
                print("---|------")
                for user in users:
                    print(f"{user['id']} | {user['name']}")
        else:
            print(f"Erro: {response.json().get('error', 'Erro')}")
            
    except Exception as e:
        print(f"Erro na requisicao: {e}")

def delete_user():
    """RF-A07: Fluxo de deletar usuario."""
    print("\n--- 3. Deletar Usuario ---")
    try:
        user_id = input("Digite o ID do usuario a ser deletado: ")
        if not user_id.isdigit():
            print("ID invalido.")
            return
            
        headers = get_auth_headers()
        url = f"{BASE_URL}/admin/users/{user_id}"
        
        response = requests.delete(url, headers=headers)
        
        if response.status_code == 200:
            print(f"Sucesso! Usuario ID {user_id} deletado.")
        else:
            print(f"Erro: {response.json().get('error', 'Usuario nao encontrado ou erro')}")

    except Exception as e:
        print(f"Erro na requisicao: {e}")

def main():
    """Loop principal do menu."""
    if not login():
        return # Encerra se o login falhar
        
    while True:
        print("\n--- Menu do Cliente Admin ---")
        print("1. Adicionar Usuario")
        print("2. Listar Usuarios")
        print("3. Deletar Usuario")
        print("4. Sair")
        choice = input("Escolha uma opcao (1-4): ")
        
        if choice == '1':
            add_user()
        elif choice == '2':
            list_users()
        elif choice == '3':
            delete_user()
        elif choice == '4':
            print("Saindo...")
            break
        else:
            print("Opcao invalida. Tente novamente.")

if __name__ == "__main__":
    main()