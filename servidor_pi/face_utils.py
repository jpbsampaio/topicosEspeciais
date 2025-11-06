import numpy as np
from deepface import DeepFace

# Define o modelo e um limiar de similaridade.
# Limiares são altamente dependentes do modelo.
# Para VGG-Face com cosseno, ~0.80 é um bom começo.
# Para FaceNet com L2, seria uma *distância* (ex: < 1.0).
MODEL_NAME = 'VGG-Face'
METRIC = 'cosine'
THRESHOLD = 0.80 

def generate_embedding(image_np):
    """
    Gera um embedding facial a partir de uma imagem (array numpy).
    Levanta ValueError se nenhum rosto ou múltiplos rostos forem detectados.
    (Implementa RNF-05)
    """
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_np,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend='opencv'
        )
        
        if len(embedding_objs) > 1:
            raise ValueError("Multiplos rostos detectados. Apenas um eh permitido.")
        
        return np.array(embedding_objs[0]['embedding'], dtype=np.float32)

    except Exception as e:
        if 'Face could not be detected' in str(e) or 'No face detected' in str(e):
            raise ValueError("Nenhum rosto detectado na imagem.")
        raise ValueError(f"Erro no processamento facial: {str(e)}")


def find_match(new_embedding, encodings_list):
    """
    Encontra a melhor correspondência para um novo embedding em uma lista de
    embeddings conhecidos (do banco de dados).
    
    'encodings_list' é uma lista de dicts: 
    [{'user_id': 1, 'name': 'Ana', 'encoding': <vetor_numpy>},...]
    
    Usa numpy para comparação 1:N vetorizada (RF-S05).
    """
    if not encodings_list:
        return None

    known_embeddings_matrix = np.array(
        [item['encoding'] for item in encodings_list]
    )
    
    user_data = [
        {"user_id": item['user_id'], "name": item['name']} 
        for item in encodings_list
    ]

    new_emb_norm = new_embedding / np.linalg.norm(new_embedding)
    
    known_emb_norm = known_embeddings_matrix / np.linalg.norm(
        known_embeddings_matrix, axis=1, keepdims=True
    )

    similarities = np.dot(known_emb_norm, new_emb_norm)
    
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]

    print(f"Melhor pontuacao: {best_score}")

    if best_score >= THRESHOLD:
        return user_data[best_match_index]
    else:
        return None # Acesso Negado
