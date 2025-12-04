"""
Cliente Kiosk para verificação de acesso por reconhecimento facial.
"""
import requests
import cv2
import numpy as np
import os

# Configuração do servidor - ALTERE PARA O IP DO SEU RASPBERRY PI
BASE_URL = os.environ.get("SERVER_URL", "http://localhost:5000")
VERIFY_URL = f"{BASE_URL}/verify"

# Tempo de exibição do resultado (em milissegundos)
DISPLAY_TIME_MS = 3000

# Cores para feedback visual
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


def draw_result_overlay(frame, access_granted, user_name=""):
    """Desenha overlay com resultado da verificação."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    if access_granted:
        # Fundo verde semi-transparente
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_GREEN, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Texto de sucesso
        cv2.putText(frame, "ACESSO LIBERADO", (w//2 - 180, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
        cv2.putText(frame, f"Bem-vindo(a), {user_name}!", (w//2 - 150, h//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
    else:
        # Fundo vermelho semi-transparente
        cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Texto de negação
        cv2.putText(frame, "ACESSO NEGADO", (w//2 - 160, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
    
    return frame


def draw_processing_overlay(frame):
    """Desenha overlay de processamento."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "Verificando...", (w//2 - 100, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
    
    return frame
def verify_face(frame):
    """Envia frame para o servidor e processa resposta."""
    
    ret, img_encoded = cv2.imencode('.jpg', frame)
    if not ret:
        print("✗ Erro ao codificar imagem.")
        return None, None

    files = {'image': ('kiosk.jpg', img_encoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post(VERIFY_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            access_granted = data.get('acesso_liberado', False)
            user_name = data.get('nome', 'Desconhecido')
            
            if access_granted:
                print(f"✓ Acesso Liberado! Bem-vindo(a), {user_name}!")
            else:
                print("✗ Acesso Negado.")
            
            return access_granted, user_name
        else:
            print(f"✗ Erro do servidor: {response.status_code}")
            return False, None

    except requests.ConnectionError:
        print(f"✗ Erro: Não foi possível conectar ao servidor em {VERIFY_URL}")
        return False, None
    except requests.Timeout:
        print("✗ Erro: Timeout na requisição.")
        return False, None


def main():
    """Loop principal do Kiosk."""
    print("\n" + "="*50)
    print("  KIOSK DE ACESSO - RECONHECIMENTO FACIAL")
    print("="*50)
    print(f"Servidor: {BASE_URL}")
    print("Pressione ESPAÇO para verificar | ESC para sair")
    print("="*50 + "\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Erro: Não foi possível abrir a webcam.")
        return
    
    # Configura resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    ret, frame = cap.read()
    if not ret:
        print("✗ Erro ao ler frame inicial.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Desenha instruções na tela
        display = frame.copy()
        cv2.putText(display, "ESPACO: Verificar | ESC: Sair", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
        cv2.imshow('Kiosk de Acesso', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # ESPAÇO
            # Mostra overlay de processamento
            processing_frame = draw_processing_overlay(frame.copy())
            cv2.imshow('Kiosk de Acesso', processing_frame)
            cv2.waitKey(1)
            
            # Verifica face
            access_granted, user_name = verify_face(frame)
            
            if access_granted is not None:
                # Mostra resultado
                result_frame = draw_result_overlay(frame.copy(), access_granted, user_name or "")
                cv2.imshow('Kiosk de Acesso', result_frame)
                cv2.waitKey(DISPLAY_TIME_MS)
            
        elif key == 27:  # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nKiosk encerrado.")


if __name__ == "__main__":
    main()
