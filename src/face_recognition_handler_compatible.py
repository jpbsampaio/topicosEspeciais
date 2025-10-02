#!/usr/bin/env python3
"""Face Recognition Handler (versão enxuta, somente OpenCV + LBPH)

Objetivo:
    Fornecer operações de coleta, treinamento e predição de faces usando apenas OpenCV.
    Remove totalmente dependências e caminhos para dlib / face_recognition.

Pipeline:
    1. Detecção: Haar Cascade frontal padrão (cv2.data.haarcascades)
    2. Dataset: imagens em tons de cinza salvas em data/<nome>/*.jpg
    3. Treino: LBPHFaceRecognizer (opencv-contrib) → models/opencv_lbph.xml
    4. Labels: mapeamento persistido em models/lbph_labels.json

Limiar:
    A saída bruta do LBPH é uma distância (menor = melhor). O valor LBPH_THRESHOLD define
    o corte para considerar uma predição como "conhecida". Ajuste conforme qualidade do dataset.
"""

from __future__ import annotations
import cv2
import numpy as np
import os
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import json
import time
from config import MODELS_DIR, DATA_DIR, LBPH_THRESHOLD


class FaceRecognitionHandler:
    """Handler principal de faces usando somente OpenCV (detecção + LBPH)."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        # Estrutura de arquivos
        self.training_data_dir = DATA_DIR
        self.lbph_model_file = os.path.join(self.models_dir, "opencv_lbph.xml")
        self.lbph_labels_file = os.path.join(self.models_dir, "lbph_labels.json")
        self.lbph_threshold = LBPH_THRESHOLD

        # Estado de labels
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}

        # Detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            self.logger.error("Falha ao carregar Haar Cascade para faces")

        # Reconhecedor LBPH
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            self.logger.error("opencv-contrib não disponível: instale opencv-contrib-python")
            self.recognizer = None

        self._load_lbph_model()
            
    # ===== Dataset & Persistência =====
        
    def load_known_faces(self) -> bool:
        """Compatibilidade: não há encodings em memória; dataset é baseado em arquivos."""
        # Nada para fazer além de tentar carregar modelo LBPH
        self._load_lbph_model()
        return True
            
    def save_known_faces(self) -> bool:
        """Mantido por compatibilidade (sem conteúdo adicional)."""
        return True
            
    def add_known_face(self, name: str, image_data: str) -> bool:
        """Decodifica imagem, detecta face, salva recorte cinza em data/<name>/ e (opcional) re-treina."""
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            if image_array.ndim == 2:  # grayscale
                bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                # Pillow dá RGB; convertemos para BGR
                bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            face_roi = self._extract_face_region(bgr)
            if face_roi is None:
                self.logger.warning(f"Nenhuma face detectada para {name}")
                return False
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            person_dir = os.path.join(self.training_data_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            filename = os.path.join(person_dir, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, gray)
            self.logger.info(f"Imagem salva para {name}: {filename}")
            if self.recognizer is not None:
                self.train_model()
            return True
        except Exception as e:
            self.logger.error(f"Erro ao adicionar face {name}: {e}")
            return False
            
    # ===== Detecção & Predição =====
        
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return [(x, y, w, h) for x, y, w, h in faces]
        except Exception as e:
            self.logger.error(f"Detecção falhou: {e}")
            return []
            
    def recognize_faces(self, frame: np.ndarray) -> Dict[str, Any]:
        """API de reconhecimento (pré-treino ou sem identificar pessoas)."""
        return self.predict(frame)
            
    # ===== Treino & Predição LBPH =====

    # ======= API pública solicitada para modo compatível (sem dlib) =======
    def train_model(self) -> bool:
        return self._train_lbph_from_dataset()

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Executa detecção e tentativa de identificação.
        Retorna lista de dicts: {name, confidence, bbox}
        - name = 'Desconhecido' se distância > limiar ou modelo não treinado.
        """
        results: list[dict] = []
        if frame is None:
            return results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            name = "Desconhecido"
            confidence = None
            distance = None
            if self._is_trained():
                try:
                    pred_label, dist = self.recognizer.predict(roi)
                    distance = float(dist)
                    confidence = distance  # mantemos chave 'confidence' como distância bruta (compat)
                    label_name = self._label_name(pred_label)
                    # Regra de aceitação: distância <= limiar E label conhecido
                    if label_name is not None and distance <= self.lbph_threshold:
                        name = label_name
                except Exception as e:
                    self.logger.error(f"Erro na predição LBPH: {e}")

            results.append({
                "name": name,
                "confidence": confidence,
                "bbox": (int(x), int(y), int(w), int(h)),
                "distance": distance,
            })
        return results

    def _label_name(self, idx: int) -> str | None:
        # Usa mapeamento label_to_name carregado do modelo
        return self.label_to_name.get(idx)

    def _is_trained(self) -> bool:
        return (
            self.recognizer is not None
            and bool(self.label_to_name)
            and os.path.exists(self.lbph_model_file)
        )

    # ===== Utilidades =====
        
    def draw_face_rectangles(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        try:
            out = frame.copy()
            for name, coords in zip(result.get('faces', []), result.get('coordinates', [])):
                top, right, bottom, left = coords['top'], coords['right'], coords['bottom'], coords['left']
                color = (0, 255, 0) if name != 'Desconhecido' else (0, 0, 255)
                cv2.rectangle(out, (left, top), (right, bottom), color, 2)
                cv2.putText(out, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return out
        except Exception:
            return frame
    
    def get_known_faces_list(self) -> List[str]:
        return sorted(set(self.label_to_name.values())) if self.label_to_name else []
        
    def get_faces_count(self) -> int:
        return len(self.get_known_faces_list())
        
    def remove_known_face(self, name: str) -> bool:
        try:
            person_dir = os.path.join(self.training_data_dir, name)
            if os.path.isdir(person_dir):
                for f in os.listdir(person_dir):
                    try:
                        os.remove(os.path.join(person_dir, f))
                    except Exception:
                        pass
                try:
                    os.rmdir(person_dir)
                except Exception:
                    pass
            if self.recognizer is not None:
                self.train_model()
            return True
        except Exception as e:
            self.logger.error(f"Erro ao remover face {name}: {e}")
            return False
            
    def clear_all_faces(self) -> bool:
        try:
            if os.path.isdir(self.training_data_dir):
                for root, dirs, files in os.walk(self.training_data_dir, topdown=False):
                    for f in files:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
                    if root != self.training_data_dir:
                        try:
                            os.rmdir(root)
                        except Exception:
                            pass
            for f in (self.lbph_model_file, self.lbph_labels_file):
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
            self.label_to_name.clear()
            self.name_to_label.clear()
            if self.recognizer is not None:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            return True
        except Exception as e:
            self.logger.error(f"Erro ao limpar faces: {e}")
            return False

    def _extract_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        faces = self._detect_faces(image)
        if not faces:
            return None
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]

    def _train_lbph_from_dataset(self) -> bool:
        if self.recognizer is None:
            return False
        images: List[np.ndarray] = []
        labels: List[int] = []
        label_to_name: Dict[int, str] = {}
        name_to_label: Dict[str, int] = {}
        os.makedirs(self.training_data_dir, exist_ok=True)
        current = 0
        for name in sorted(os.listdir(self.training_data_dir)):
            pdir = os.path.join(self.training_data_dir, name)
            if not os.path.isdir(pdir):
                continue
            name_to_label[name] = current
            label_to_name[current] = name
            for file in os.listdir(pdir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(pdir, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None or img.size == 0:
                        continue
                    images.append(img)
                    labels.append(current)
            current += 1
        if not images:
            self.logger.warning("Sem imagens para treino LBPH")
            return False
        try:
            self.recognizer.train(images, np.array(labels))
            self.recognizer.write(self.lbph_model_file)
            with open(self.lbph_labels_file, 'w', encoding='utf-8') as f:
                json.dump({"label_to_name": label_to_name, "name_to_label": name_to_label}, f, ensure_ascii=False)
            self.label_to_name = label_to_name
            self.name_to_label = name_to_label
            return True
        except Exception as e:
            self.logger.error(f"Treino LBPH falhou: {e}")
            return False

    def _load_lbph_model(self) -> None:
        if self.recognizer is None:
            return
        try:
            if os.path.exists(self.lbph_model_file) and os.path.exists(self.lbph_labels_file):
                self.recognizer.read(self.lbph_model_file)
                with open(self.lbph_labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.label_to_name = {int(k): v for k, v in data.get('label_to_name', {}).items()}
                self.name_to_label = {k: int(v) for k, v in data.get('name_to_label', {}).items()}
                self.logger.info("Modelo LBPH carregado")
        except Exception as e:
            self.logger.error(f"Falha ao carregar modelo LBPH: {e}")
