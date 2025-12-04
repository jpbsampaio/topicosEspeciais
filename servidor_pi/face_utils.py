"""
Módulo de reconhecimento facial usando OpenCV LBPH.
Otimizado para Raspberry Pi 3 com recursos limitados.
"""
import cv2
import numpy as np
import os
import pickle

# Configurações do LBPH
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

# Limiar de confiança - quanto MENOR o valor, melhor a correspondência
# Valores típicos: 50-80 (ajuste conforme necessário)
LBPH_THRESHOLD = 80.0

# Caminho para o modelo LBPH treinado
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lbph_model.yml')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.pkl')

# Detector de faces Haar Cascade
# Compatível com diferentes versões do OpenCV
def _get_cascade_path():
    """Obtém o caminho do Haar Cascade de forma compatível."""
    # Tenta usar cv2.data (OpenCV 4.x via pip)
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        return cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Caminhos comuns no Raspberry Pi OS
    possible_paths = [
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fallback: tenta encontrar com locate
    raise FileNotFoundError(
        "Haar Cascade não encontrado. Instale com: sudo apt install opencv-data"
    )

CASCADE_PATH = _get_cascade_path()
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Modelo LBPH global
_recognizer = None
_label_map = {}  # {label_id: {'user_id': int, 'name': str}}


def _ensure_model_dir():
    """Garante que o diretório de modelos existe."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def _get_recognizer():
    """Retorna o reconhecedor LBPH, carregando do disco se necessário."""
    global _recognizer, _label_map
    
    if _recognizer is None:
        _recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=LBPH_RADIUS,
            neighbors=LBPH_NEIGHBORS,
            grid_x=LBPH_GRID_X,
            grid_y=LBPH_GRID_Y
        )
        
        # Tenta carregar modelo existente
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            try:
                _recognizer.read(MODEL_PATH)
                with open(LABELS_PATH, 'rb') as f:
                    _label_map = pickle.load(f)
                print(f"Modelo LBPH carregado com {len(_label_map)} usuários.")
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
                _label_map = {}
    
    return _recognizer


def detect_face(image_np):
    """
    Detecta uma face na imagem e retorna a região recortada em escala de cinza.
    Levanta ValueError se nenhuma face ou múltiplas faces forem detectadas.
    
    Args:
        image_np: Imagem BGR (numpy array)
    
    Returns:
        face_gray: Região da face em escala de cinza, redimensionada para 200x200
    """
    if image_np is None:
        raise ValueError("Imagem inválida.")
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Equaliza histograma para melhorar contraste
    gray = cv2.equalizeHist(gray)
    
    # Detecta faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        raise ValueError("Nenhum rosto detectado na imagem.")
    
    if len(faces) > 1:
        raise ValueError("Múltiplos rostos detectados. Apenas um é permitido.")
    
    # Extrai a região da face
    x, y, w, h = faces[0]
    face_gray = gray[y:y+h, x:x+w]
    
    # Redimensiona para tamanho padrão
    face_gray = cv2.resize(face_gray, (200, 200))
    
    return face_gray


def add_face_to_model(user_id, name, image_np):
    """
    Adiciona uma nova face ao modelo LBPH.
    
    Args:
        user_id: ID do usuário no banco de dados
        name: Nome do usuário
        image_np: Imagem BGR contendo a face
    
    Returns:
        label: O label atribuído a este usuário
    """
    global _label_map
    
    _ensure_model_dir()
    recognizer = _get_recognizer()
    
    # Detecta e processa a face
    face_gray = detect_face(image_np)
    
    # Verifica se usuário já tem um label
    existing_label = None
    for label, data in _label_map.items():
        if data['user_id'] == user_id:
            existing_label = label
            break
    
    if existing_label is not None:
        label = existing_label
    else:
        # Novo label é o próximo número disponível
        label = max(_label_map.keys(), default=-1) + 1
        _label_map[label] = {'user_id': user_id, 'name': name}
    
    # Prepara dados para treinamento
    faces = [face_gray]
    labels = np.array([label])
    
    # Treina ou atualiza o modelo
    if os.path.exists(MODEL_PATH):
        # Atualiza modelo existente
        recognizer.update(faces, labels)
    else:
        # Cria novo modelo
        recognizer.train(faces, labels)
    
    # Salva modelo e labels
    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(_label_map, f)
    
    print(f"Face adicionada para '{name}' (user_id={user_id}, label={label})")
    
    return label


def recognize_face(image_np):
    """
    Reconhece uma face na imagem.
    
    Args:
        image_np: Imagem BGR contendo a face
    
    Returns:
        dict com 'user_id', 'name', 'confidence' se reconhecido
        None se não reconhecido ou abaixo do limiar
    """
    global _label_map
    
    recognizer = _get_recognizer()
    
    # Verifica se há modelo treinado
    if not _label_map:
        print("Nenhum modelo treinado disponível.")
        return None
    
    try:
        face_gray = detect_face(image_np)
    except ValueError as e:
        raise e
    
    # Realiza predição
    label, confidence = recognizer.predict(face_gray)
    
    print(f"Predição: label={label}, confiança={confidence:.2f} (limiar={LBPH_THRESHOLD})")
    
    # Verifica limiar (LBPH usa distância, menor é melhor)
    if confidence <= LBPH_THRESHOLD:
        if label in _label_map:
            user_data = _label_map[label]
            return {
                'user_id': user_data['user_id'],
                'name': user_data['name'],
                'confidence': confidence
            }
    
    return None


def remove_user_from_model(user_id):
    """
    Remove um usuário do modelo.
    Nota: LBPH não suporta remoção incremental, então precisamos
    reconstruir o modelo sem este usuário.
    
    Args:
        user_id: ID do usuário a remover
    """
    global _label_map, _recognizer
    
    # Remove do mapa de labels
    label_to_remove = None
    for label, data in _label_map.items():
        if data['user_id'] == user_id:
            label_to_remove = label
            break
    
    if label_to_remove is not None:
        del _label_map[label_to_remove]
        
        # Salva labels atualizados
        _ensure_model_dir()
        with open(LABELS_PATH, 'wb') as f:
            pickle.dump(_label_map, f)
        
        print(f"Usuário {user_id} removido do modelo (label={label_to_remove})")
        
        # Se não há mais usuários, remove o modelo
        if not _label_map:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            _recognizer = None
            print("Modelo removido (sem usuários restantes).")


def get_model_stats():
    """Retorna estatísticas do modelo atual."""
    _get_recognizer()  # Garante que o modelo está carregado
    
    return {
        'total_users': len(_label_map),
        'users': [
            {'user_id': data['user_id'], 'name': data['name']}
            for data in _label_map.values()
        ],
        'threshold': LBPH_THRESHOLD,
        'model_exists': os.path.exists(MODEL_PATH)
    }


def clear_model():
    """Remove completamente o modelo e labels."""
    global _recognizer, _label_map
    
    _recognizer = None
    _label_map = {}
    
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(LABELS_PATH):
        os.remove(LABELS_PATH)
    
    print("Modelo LBPH limpo.")
