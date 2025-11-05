# cliente_acesso/kiosk.py
import requests
import cv2
import numpy as np

# --- Configuracao ---
# CORREÇÃO: 127.0.0.1 é localhost. Para funcionar em outra máquina,
# você DEVE colocar o IP real do seu Raspberry Pi na rede.
BASE_URL = "http://<IP_DO_SEU_RASPBERRY_PI>:5000" 
VERIFY_URL = f"{BASE_URL}/verify"
DISPLAY_TIME_MS = 5000 # RNF-03: 5 segundos

# --- Carregar Assets (RF-C05) ---
try:
    IMG_LOCKED = cv2.imread('cliente_acesso/assets/locked.png')
    IMG_UNLOCKED = cv2.imread('cliente_acesso/assets/unlocked.png')
    # CORREÇÃO: A checagem de falha do imread deve ser 'is None'
    if IMG_LOCKED is None or IMG_UNLOCKED is None:
        raise IOError("Nao foi possivel carregar imagens de assets.")
except Exception as e:
    print(f"Erro ao carregar assets: {e}")
    print("Simulacao de tranca sera desabilitada.")
    IMG_LOCKED = None
    IMG_UNLOCKED = None

def show_lock_status(access_granted):
    """RF-C04, RF-C05: Exibe o status da tranca por 5 segundos."""
    # CORREÇÃO: A checagem de falha do imread deve ser 'is None'
    if IMG_LOCKED is None or IMG_UNLOCKED is None:
        print("Assets nao carregados. Pulando exibicao.")
        return

    window_name = 'Status da Tranca'
    
    if access_granted:
        cv2.imshow(window_name, IMG_UNLOCKED)
    else:
        cv2.imshow(window_name, IMG_LOCKED)
        
    # RNF-03: Mantem a janela visivel por 5 segundos
    cv2.waitKey(DISPLAY_TIME_MS)
    
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass # Janela pode ter sido fechada manualmente

def verify_face(frame):
    """RF-C03: Captura, envia e lida com a resposta."""
    
    # Codifica a imagem para JPEG em memoria
    ret, img_encoded = cv2.imencode('.jpg', frame)
    if not ret:
        print("Erro ao codificar imagem.")
        return

    # Prepara o payload multipart/form-data
    files = {'image': ('kiosk.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(VERIFY_URL, files=files, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            access_granted = data.get('acesso_liberado', False)
            user_name = data.get('nome', 'Desconhecido')
            
            if access_granted:
                print(f"Acesso Liberado! Bem-vindo(a), {user_name}!")
            else:
                print("Acesso Negado.")
            
            # Chama a simulacao de tranca
            show_lock_status(access_granted)
            
        else:
            print(f"Erro do servidor: {response.status_code}")
            show_lock_status(False)

    except requests.ConnectionError:
        print(f"Erro: Nao foi possivel conectar ao servidor em {VERIFY_URL}.")
    except requests.Timeout:
        print("Erro: A requisicao expirou (timeout).")
        
def main():
    """RF-C01: Loop principal do Kiosk."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Nao foi possivel abrir a webcam.")
        return
        
    # CORREÇÃO: Pega as dimensões do frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame inicial.")
        cap.release()
        return
        
    H, W = frame.shape[:2]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Desenha o texto de instrucao
        cv2.putText(frame, "Pressione ESPACO para verificar", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Kiosk de Acesso', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # ESPACO (RF-C02)
            print("Verificando...")
            
            # Feedback imediato na tela (RNF-04)
            overlay = frame.copy()
            # CORREÇÃO: As coordenadas do retângulo usam (largura, altura)
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # CORREÇÃO: As coordenadas do texto usam (x, y). 
            # frame.shape[1] é Largura (W), frame.shape[0] é Altura (H)
            cv2.putText(frame, "Verificando...", (W//2 - 100, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Kiosk de Acesso', frame)
            cv2.waitKey(1) # Atualiza a janela
            
            # Envia o frame original (sem o texto "Verificando") para verificacao
            # (Nota: o frame enviado aqui é o *antes* do overlay)
            verify_face(overlay) # Envia o frame original antes do overlay
            
        elif key == 27: # ESC para sair
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()