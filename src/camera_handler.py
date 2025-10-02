#!/usr/bin/env python3
"""
Handler para Gerenciamento da Câmera

Responsabilidades principais:
- Abrir/fechar a câmera (OpenCV VideoCapture) e configurar resolução.
- Capturar frames sob demanda (capture_frame) ou continuamente em uma thread separada.
- Manter um buffer (queue) com poucos frames recentes para reduzir latência.
- Codificar frames (JPEG/PNG) para transmissão via socket.

Por que usar queue e locks?
- A captura contínua roda em outra thread para não bloquear as requisições do servidor.
- Usamos uma Queue pequena (maxsize=5) para evitar atraso (latência) — descartamos frames antigos.
- Um Lock protege o frame atual durante leitura/gravação entre threads.
"""

import cv2
import numpy as np
import logging
import threading
import time
from typing import Optional, Tuple, Any
import queue


class CameraHandler:
    """Gerencia a captura de vídeo da câmera."""
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Inicializa o handler da câmera.
        
        Args:
            camera_index: Índice da câmera (0 para câmera padrão)
            resolution: Resolução desejada (largura, altura)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self.is_recording = False
        
        # Threading para captura contínua
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=5)  # buffer curto = menor latência
        self.stop_capture = threading.Event()
        
        # Frame atual
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        
        # Estatísticas
        self.frames_captured = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def initialize_camera(self) -> bool:
        """
        Inicializa a câmera.
        
        Returns:
            True se inicializou com sucesso, False caso contrário
        """
        try:
            self.logger.info(f"Inicializando câmera {self.camera_index}")
            
            # Inicializa captura de vídeo
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Falha ao abrir câmera {self.camera_index}")
                return False
            
            # Configura resolução
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Configura buffer interno do OpenCV para reduzir latência
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Testa captura
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Falha ao capturar frame de teste")
                return False
            
            self.is_initialized = True
            self.logger.info(f"Câmera inicializada com resolução {frame.shape[1]}x{frame.shape[0]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar câmera: {e}")
            return False
            
    def start_continuous_capture(self) -> bool:
        """
        Inicia captura contínua em thread separada.
        
        Returns:
            True se iniciou com sucesso, False caso contrário
        """
        if not self.is_initialized:
            self.logger.error("Câmera não inicializada")
            return False
            
        if self.is_recording:
            self.logger.warning("Captura contínua já está ativa")
            return True
            
        try:
            self.stop_capture.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.is_recording = True
            self.logger.info("Captura contínua iniciada")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar captura contínua: {e}")
            return False
            
    def stop_continuous_capture(self) -> None:
        """Para a captura contínua."""
        if self.is_recording:
            self.stop_capture.set()
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                
            self.is_recording = False
            self.logger.info("Captura contínua parada")
            
    def _capture_loop(self) -> None:
        """Loop principal de captura (executa em thread separada)."""
        self.logger.info("Iniciando loop de captura")
        
        while not self.stop_capture.is_set():  # loop até receber sinal de parada
            try:
                if not self.cap or not self.cap.isOpened():
                    self.logger.error("Câmera não disponível no loop de captura")
                    break
                    
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Atualiza frame atual
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        
                    # Adiciona à queue (se cheia, remove o mais antigo para manter buffer curto)
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        try:
                            self.frame_queue.get(block=False)  # Remove frame antigo
                            self.frame_queue.put(frame, block=False)
                        except queue.Empty:
                            pass
                            
                    # Atualiza estatísticas
                    self.frames_captured += 1
                    self._update_fps()
                    
                else:
                    self.logger.warning("Falha ao capturar frame")
                    time.sleep(0.1)  # Pausa breve em caso de erro
                    
            except Exception as e:
                self.logger.error(f"Erro no loop de captura: {e}")
                time.sleep(0.1)
                
        self.logger.info("Loop de captura finalizado")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Captura um frame da câmera.
        
        Returns:
            Frame capturado ou None se falhou
        """
        try:
            if not self.is_initialized or not self.cap:
                self.logger.error("Câmera não inicializada")
                return None
                
            if self.is_recording:
                # Se captura contínua está ativa, usa frame atual
                with self.frame_lock:
                    if self.current_frame is not None:
                        return self.current_frame.copy()
                    else:
                        self.logger.warning("Nenhum frame disponível na captura contínua")
                        return None
            else:
                # Captura direta
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    self.frames_captured += 1
                    return frame
                else:
                    self.logger.error("Falha ao capturar frame")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao capturar frame: {e}")
            return None
            
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Obtém o frame mais recente da queue.
        
        Returns:
            Frame mais recente ou None se não disponível
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
            
    def encode_frame(self, frame: np.ndarray, format: str = '.jpg', quality: int = 85) -> Tuple[bool, np.ndarray]:
        """
        Codifica frame para transmissão.
        
        Args:
            frame: Frame a ser codificado
            format: Formato de codificação (.jpg, .png)
            quality: Qualidade da compressão (1-100 para JPEG)
            
        Returns:
            Tupla (sucesso, dados_codificados)
        """
        try:
            if format.lower() == '.jpg' or format.lower() == '.jpeg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format.lower() == '.png':
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                encode_params = []
                
            success, encoded_img = cv2.imencode(format, frame, encode_params)
            
            if success:
                return True, encoded_img
            else:
                self.logger.error(f"Falha ao codificar frame no formato {format}")
                return False, np.array([])
                
        except Exception as e:
            self.logger.error(f"Erro ao codificar frame: {e}")
            return False, np.array([])
            
    def resize_frame(self, frame: np.ndarray, width: int = None, height: int = None, 
                    scale_factor: float = None) -> np.ndarray:
        """
        Redimensiona um frame.
        
        Args:
            frame: Frame original
            width: Nova largura
            height: Nova altura
            scale_factor: Fator de escala (alternativa a width/height)
            
        Returns:
            Frame redimensionado
        """
        try:
            if scale_factor is not None:
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
            elif width is not None and height is not None:
                new_width = width
                new_height = height
            elif width is not None:
                # Mantém proporção baseada na largura
                ratio = width / frame.shape[1]
                new_width = width
                new_height = int(frame.shape[0] * ratio)
            elif height is not None:
                # Mantém proporção baseada na altura
                ratio = height / frame.shape[0]
                new_width = int(frame.shape[1] * ratio)
                new_height = height
            else:
                return frame  # Sem redimensionamento
                
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_frame
            
        except Exception as e:
            self.logger.error(f"Erro ao redimensionar frame: {e}")
            return frame
            
    def apply_filters(self, frame: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
        """
        Aplica filtros básicos ao frame.
        
        Args:
            frame: Frame original
            brightness: Ajuste de brilho (-100 a 100)
            contrast: Ajuste de contraste (0.5 a 2.0)
            
        Returns:
            Frame com filtros aplicados
        """
        try:
            # Aplica ajustes
            adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            return adjusted
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar filtros: {e}")
            return frame
            
    def _update_fps(self) -> None:
        """Atualiza contador de FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # A cada segundo
            self.logger.debug(f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.last_fps_time = current_time
            
    def get_camera_info(self) -> dict:
        """
        Retorna informações da câmera.
        
        Returns:
            Dicionário com informações da câmera
        """
        if not self.is_initialized or not self.cap:
            return {"error": "Câmera não inicializada"}
            
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                "camera_index": self.camera_index,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "is_recording": self.is_recording,
                "frames_captured": self.frames_captured,
                "queue_size": self.frame_queue.qsize()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter informações da câmera: {e}")
            return {"error": str(e)}
            
    def cleanup(self) -> None:
        """Limpa recursos da câmera."""
        self.logger.info("Limpando recursos da câmera")
        
        # Para captura contínua
        self.stop_continuous_capture()
        
        # Fecha câmera
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Limpa queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except queue.Empty:
                break
                
        self.is_initialized = False
        self.logger.info("Recursos da câmera liberados")
        
    def __del__(self):
        """Destrutor para garantir limpeza de recursos."""
        self.cleanup()
