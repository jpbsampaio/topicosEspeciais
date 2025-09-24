#!/usr/bin/env python3
"""
Servidor de Reconhecimento Facial
Implementa arquitetura cliente-servidor com ThreadPool para conexões simultâneas.
"""

import socket
import threading
import logging
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import time
import argparse
import os

from camera_handler import CameraHandler
from face_recognition_handler_compatible import FaceRecognitionHandler
from config import (
    SERVER_HOST,
    SERVER_PORT,
    MAX_WORKERS,
    LOG_DIR,
    DATA_DIR,
    CAMERA_INDEX,
    camera_resolution,
    LBPH_THRESHOLD,
)


class FacialRecognitionServer:
    """Servidor principal para reconhecimento facial com suporte a múltiplos clientes."""
    
    def __init__(self, host: str = 'localhost', port: int = 8888, max_workers: int = 5,
                 camera_index: int = CAMERA_INDEX, resolution: tuple = None):
        """
        Inicializa o servidor.
        
        Args:
            host: Endereço IP do servidor
            port: Porta do servidor
            max_workers: Número máximo de threads no pool
        """
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server_socket: Optional[socket.socket] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False
        
        # Handlers especializados
        self.face_handler = FaceRecognitionHandler()
        if resolution is None:
            resolution = camera_resolution()
        self.camera_handler = CameraHandler(camera_index=camera_index, resolution=resolution)
        
        # Controle de conexões ativas
        self.active_connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        
        # Configuração de logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configura o sistema de logging."""
        # Garante diretório de logs
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
        except Exception:
            pass

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(LOG_DIR, 'server.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start_server(self) -> None:
        """Inicia o servidor e aceita conexões."""
        try:
            # Configuração do socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            # Inicialização do ThreadPool
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            self.is_running = True
            self.logger.info(f"Servidor iniciado em {self.host}:{self.port}")
            self.logger.info(f"ThreadPool configurado com {self.max_workers} workers")
            
            # Inicializa handlers
            ok = self.camera_handler.initialize_camera()
            if not ok:
                self.logger.error("Falha ao inicializar a câmera. Verifique:")
                self.logger.error("1) Se outra aplicação está usando a webcam")
                self.logger.error("2) Configurações do Windows: Privacidade e segurança > Câmera > Permitir acesso para aplicativos desktop")
                self.logger.error("3) Teste a webcam no aplicativo Câmera do Windows")
            self.face_handler.load_known_faces()
            
            # Loop principal de aceitação de conexões
            while self.is_running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.logger.info(f"Nova conexão de {client_address}")
                    
                    # Submete a conexão para o ThreadPool
                    future = self.executor.submit(self.handle_client, client_socket, client_address)
                    
                    # Adiciona callback para limpeza quando a conexão terminar
                    future.add_done_callback(lambda f: self._cleanup_connection(client_address))
                    
                except socket.error as e:
                    if self.is_running:
                        self.logger.error(f"Erro ao aceitar conexão: {e}")
                        
        except Exception as e:
            self.logger.error(f"Erro ao iniciar servidor: {e}")
        finally:
            self.shutdown()
            
    def handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        """
        Gerencia a comunicação com um cliente específico.
        
        Args:
            client_socket: Socket do cliente
            client_address: Endereço do cliente
        """
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        try:
            # Registra a conexão
            with self.connection_lock:
                self.active_connections[client_id] = client_socket
                
            self.logger.info(f"Iniciando atendimento ao cliente {client_id}")
            
            # Envia mensagem de boas-vindas
            welcome_msg = {
                "type": "welcome",
                "message": "Conectado ao servidor de reconhecimento facial",
                "timestamp": time.time()
            }
            self._send_message(client_socket, welcome_msg)
            
            # Loop de comunicação com o cliente (bufferizado por linhas)
            recv_buffer = b""
            while self.is_running:
                try:
                    # Recebe dados do cliente
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    recv_buffer += data

                    # Processa todas as mensagens completas (terminadas com \n)
                    while b"\n" in recv_buffer:
                        line, recv_buffer = recv_buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            message = json.loads(line.decode('utf-8'))
                        except json.JSONDecodeError:
                            error_response = {
                                "type": "error",
                                "message": "Formato de mensagem inválido",
                                "timestamp": time.time()
                            }
                            self._send_message(client_socket, error_response)
                            continue

                        response = self._process_client_message(message)
                        self._send_message(client_socket, response)

                except socket.timeout:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Erro ao atender cliente {client_id}: {e}")
        finally:
            self._disconnect_client(client_socket, client_id)
            
    def _process_client_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens recebidas do cliente.
        
        Args:
            message: Mensagem do cliente
            
        Returns:
            Resposta para o cliente
        """
        message_type = message.get("type", "unknown")
        
        if message_type == "recognize_face":
            return self._handle_face_recognition()
            
        # 'capture_image' removido conforme nova UX
            
        elif message_type == "add_known_face":
            return self._handle_add_known_face(message)
            
        elif message_type == "list_known_faces":
            return self._handle_list_known_faces()

        elif message_type == "train_model":
            return self._handle_train_model()

        elif message_type == "clear_model":
            return self._handle_clear_model()

        elif message_type == "predict":
            return self._handle_predict()
            
        elif message_type == "collect_dataset":
            return self._handle_collect_dataset(message)

        elif message_type == "authorize_access":
            return self._handle_authorize_access(message)

        elif message_type == "ping":
            return {
                "type": "pong",
                "timestamp": time.time()
            }
            
        else:
            return {
                "type": "error",
                "message": f"Tipo de mensagem não reconhecido: {message_type}",
                "timestamp": time.time()
            }
            
    def _handle_face_recognition(self) -> Dict[str, Any]:
        """Executa reconhecimento facial."""
        try:
            # Captura frame da câmera
            frame = self.camera_handler.capture_frame()
            if frame is None:
                return {
                    "type": "error",
                    "message": "Falha ao capturar imagem da câmera",
                    "timestamp": time.time()
                }
                
            # Executa reconhecimento
            result = self.face_handler.recognize_faces(frame)
            
            # Codifica imagem para envio (opcional)
            _, buffer = self.camera_handler.encode_frame(frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "type": "recognition_result",
                "recognized_faces": result["faces"],
                "confidence_scores": result["confidence"],
                "image_data": image_data,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Erro no reconhecimento facial: {e}")
            return {
                "type": "error",
                "message": f"Erro no reconhecimento: {str(e)}",
                "timestamp": time.time()
            }
            
    # _handle_image_capture removido conforme nova UX
            
    def _handle_collect_dataset(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Coleta N imagens via câmera para o dataset data/<name>/ usando o fluxo de add_known_face."""
        try:
            name = message.get("name")
            count = int(message.get("count", 20))
            if not name or count <= 0:
                return {"type": "error", "message": "Parâmetros inválidos para coleta de dataset", "timestamp": time.time()}

            saved = 0
            attempts = 0
            max_attempts = count * 3  # tolera algumas falhas de detecção

            while saved < count and attempts < max_attempts:
                attempts += 1
                frame = self.camera_handler.capture_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                # Reaproveita pipeline existente: encode -> base64 -> add_known_face
                ok, buf = self.camera_handler.encode_frame(frame)
                if not ok:
                    continue
                b64 = base64.b64encode(buf).decode('utf-8')
                if self.face_handler.add_known_face(name, b64):
                    saved += 1
                    time.sleep(0.2)

            return {
                "type": "dataset_collected",
                "success": saved > 0,
                "requested": count,
                "saved": saved,
                "name": name,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Erro na coleta de dataset: {e}")
            return {"type": "error", "message": f"Erro na coleta: {str(e)}", "timestamp": time.time()}

    def _handle_add_known_face(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Adiciona nova face conhecida."""
        try:
            name = message.get("name")
            image_data = message.get("image_data")
            
            if not name or not image_data:
                return {
                    "type": "error",
                    "message": "Nome e dados da imagem são obrigatórios",
                    "timestamp": time.time()
                }
                
            success = self.face_handler.add_known_face(name, image_data)
            
            if success:
                return {
                    "type": "face_added",
                    "message": f"Face de {name} adicionada com sucesso",
                    "timestamp": time.time()
                }
            else:
                return {
                    "type": "error",
                    "message": "Falha ao adicionar face",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "type": "error",
                "message": f"Erro ao adicionar face: {str(e)}",
                "timestamp": time.time()
            }
            
    def _handle_list_known_faces(self) -> Dict[str, Any]:
        """Lista faces conhecidas."""
        try:
            known_faces = self.face_handler.get_known_faces_list()
            return {
                "type": "known_faces_list",
                "faces": known_faces,
                "count": len(known_faces),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "type": "error",
                "message": f"Erro ao listar faces: {str(e)}",
                "timestamp": time.time()
            }

    def _handle_train_model(self) -> Dict[str, Any]:
        """Treina o modelo (LBPH no modo compatível)."""
        try:
            success = False
            if hasattr(self.face_handler, 'train_model'):
                success = self.face_handler.train_model()
            dataset_counts, total_images = self._dataset_counts()
            return {
                "type": "model_trained",
                "success": bool(success),
                "known_faces": self.face_handler.get_known_faces_list(),
                "dataset_counts": dataset_counts,
                "total_images": total_images,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo: {e}")
            return {
                "type": "error",
                "message": f"Erro ao treinar modelo: {str(e)}",
                "timestamp": time.time()
            }

    def _handle_clear_model(self) -> Dict[str, Any]:
        """Limpa dataset/modelo e zera faces conhecidas."""
        try:
            success = self.face_handler.clear_all_faces()
            return {
                "type": "model_cleared",
                "success": bool(success),
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Erro ao limpar modelo: {e}")
            return {
                "type": "error",
                "message": f"Erro ao limpar modelo: {str(e)}",
                "timestamp": time.time()
            }

    def _handle_predict(self) -> Dict[str, Any]:
        """Executa predição (equivalente a reconhecer usando handler.predict)."""
        try:
            frame = self.camera_handler.capture_frame()
            if frame is None:
                return {
                    "type": "error",
                    "message": "Falha ao capturar imagem da câmera",
                    "timestamp": time.time()
                }

            if hasattr(self.face_handler, 'predict'):
                result = self.face_handler.predict(frame)
            else:
                result = self.face_handler.recognize_faces(frame)

            _, buffer = self.camera_handler.encode_frame(frame)
            image_data = base64.b64encode(buffer).decode('utf-8')

            return {
                "type": "prediction_result",
                "recognized_faces": result.get("faces", []),
                "confidence_scores": result.get("confidence", []),
                "image_data": image_data,
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Erro na predição: {e}")
            return {
                "type": "error",
                "message": f"Erro na predição: {str(e)}",
                "timestamp": time.time()
            }

    def _handle_authorize_access(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza votação em janela deslizante: exige 'required' acertos em 'count' frames.

        Parâmetros (com defaults):
          - count: número de frames a capturar (default 3)
          - required: votos necessários para conceder acesso (default 2)
          - threshold: limiar LBPH para aceitar predição (default LBPH_THRESHOLD)
        """
        try:
            count = int(message.get("count", 3))
            required = int(message.get("required", 2))
            threshold = float(message.get("threshold", LBPH_THRESHOLD))
            if count <= 0 or required <= 0 or required > count:
                return {"type": "error", "message": "Parâmetros inválidos (count/required)", "timestamp": time.time()}

            tallies: Dict[str, int] = {}
            frames_details = []
            last_image_b64 = None

            for i in range(count):
                frame = self.camera_handler.capture_frame()
                if frame is None:
                    frames_details.append({"error": "Falha captura"})
                    time.sleep(0.1)
                    continue

                # Executa predição
                if hasattr(self.face_handler, 'predict'):
                    result = self.face_handler.predict(frame)
                else:
                    result = self.face_handler.recognize_faces(frame)

                faces = result.get("faces", [])
                confs = result.get("confidence", [])
                accepted_this_frame = []

                # Considera apenas rótulos conhecidos com confiança abaixo do limiar
                for idx, name in enumerate(faces):
                    if not name or name == "Desconhecido":
                        continue
                    conf = None
                    if idx < len(confs):
                        try:
                            conf = float(confs[idx])
                        except Exception:
                            conf = None
                    if conf is not None and conf <= threshold:
                        tallies[name] = tallies.get(name, 0) + 1
                        accepted_this_frame.append({"name": name, "confidence": conf})

                # Codifica último frame para retorno
                _, buffer = self.camera_handler.encode_frame(frame)
                last_image_b64 = base64.b64encode(buffer).decode('utf-8')

                frames_details.append({
                    "faces": faces,
                    "confidences": confs,
                    "accepted": accepted_this_frame,
                })

                time.sleep(0.15)

            # Decide vencedor
            winner = None
            votes = 0
            if tallies:
                winner, votes = max(tallies.items(), key=lambda kv: kv[1])
            granted = bool(winner and votes >= required)

            return {
                "type": "access_decision",
                "granted": granted,
                "name": winner,
                "votes": votes,
                "required": required,
                "count": count,
                "threshold": threshold,
                "tallies": tallies,
                "frames": frames_details,
                "image_data": last_image_b64,
                "timestamp": time.time(),
            }
        except Exception as e:
            self.logger.error(f"Erro em authorize_access: {e}")
            return {
                "type": "error",
                "message": f"Erro em authorize_access: {str(e)}",
                "timestamp": time.time()
            }
            
    def _send_message(self, client_socket: socket.socket, message: Dict[str, Any]) -> None:
        """Envia mensagem para o cliente."""
        try:
            data = json.dumps(message).encode('utf-8') + b"\n"
            client_socket.sendall(data)
        except Exception as e:
            self.logger.error(f"Erro ao enviar mensagem: {e}")
            
    def _disconnect_client(self, client_socket: socket.socket, client_id: str) -> None:
        """Desconecta um cliente e limpa recursos."""
        try:
            client_socket.close()
            with self.connection_lock:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
            self.logger.info(f"Cliente {client_id} desconectado")
        except Exception as e:
            self.logger.error(f"Erro ao desconectar cliente {client_id}: {e}")
            
    def _cleanup_connection(self, client_address: tuple) -> None:
        """Callback para limpeza quando uma conexão termina."""
        client_id = f"{client_address[0]}:{client_address[1]}"
        self.logger.info(f"Limpeza da conexão {client_id} completada")
        
    def get_server_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do servidor."""
        with self.connection_lock:
            return {
                "active_connections": len(self.active_connections),
                "max_workers": self.max_workers,
                "is_running": self.is_running,
                "connected_clients": list(self.active_connections.keys())
            }

    def _dataset_counts(self) -> tuple[Dict[str, int], int]:
        """Conta imagens por pessoa no diretório de dataset."""
        counts: Dict[str, int] = {}
        total = 0
        try:
            if os.path.isdir(DATA_DIR):
                for name in os.listdir(DATA_DIR):
                    person_dir = os.path.join(DATA_DIR, name)
                    if not os.path.isdir(person_dir):
                        continue
                    c = 0
                    for f in os.listdir(person_dir):
                        lf = f.lower()
                        if lf.endswith('.jpg') or lf.endswith('.jpeg') or lf.endswith('.png'):
                            c += 1
                    if c > 0:
                        counts[name] = c
                        total += c
        except Exception as e:
            self.logger.error(f"Erro ao contar dataset: {e}")
        return counts, total
            
    def shutdown(self) -> None:
        """Encerra o servidor de forma segura."""
        self.logger.info("Iniciando shutdown do servidor...")
        self.is_running = False
        
        # Fecha todas as conexões ativas
        with self.connection_lock:
            for client_id, client_socket in self.active_connections.items():
                try:
                    client_socket.close()
                    self.logger.info(f"Conexão {client_id} fechada")
                except Exception as e:
                    self.logger.error(f"Erro ao fechar conexão {client_id}: {e}")
            self.active_connections.clear()
        
        # Encerra o ThreadPool
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("ThreadPool encerrado")
            
        # Fecha socket do servidor
        if self.server_socket:
            self.server_socket.close()
            self.logger.info("Socket do servidor fechado")
            
        # Limpa recursos dos handlers
        self.camera_handler.cleanup()
        
        self.logger.info("Shutdown completado")


def main():
    """Função principal para executar o servidor."""
    parser = argparse.ArgumentParser(description="Servidor de reconhecimento facial")
    parser.add_argument("--host", default=SERVER_HOST, help=f"Endereço do servidor (default: {SERVER_HOST})")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"Porta do servidor (default: {SERVER_PORT})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Quantidade de workers no ThreadPool (default: {MAX_WORKERS})")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help=f"Índice da câmera (default: {CAMERA_INDEX})")
    args = parser.parse_args()

    server = FacialRecognitionServer(host=args.host, port=args.port, max_workers=args.workers,
                                     camera_index=args.camera_index, resolution=camera_resolution())
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nEncerrando servidor...")
        server.shutdown()


if __name__ == "__main__":
    main()
