#!/usr/bin/env python3
"""
Cliente para Teste do Sistema de Reconhecimento Facial
Cliente interativo para comunicação com o servidor de reconhecimento facial.
"""

import socket
import json
import time
import threading
import logging
import base64
from typing import Dict, Any, Optional
import os
import argparse
import os
from config import SERVER_HOST, SERVER_PORT, LOG_DIR, LBPH_THRESHOLD


class FacialRecognitionClient:
    """Cliente para comunicação com o servidor de reconhecimento facial."""
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        """
        Inicializa o cliente.
        
        Args:
            host: Endereço do servidor
            port: Porta do servidor
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.is_connected = False
        
        # Thread para recebimento de mensagens
        self.receive_thread: Optional[threading.Thread] = None
        self.stop_receiving = threading.Event()
        # Sinalização para fluxos síncronos (coleta guiada)
        self._dataset_event = threading.Event()
        self._dataset_result: Optional[Dict[str, Any]] = None
        
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
                logging.FileHandler(os.path.join(LOG_DIR, 'client.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """
        Conecta ao servidor.
        
        Returns:
            True se conectou com sucesso, False caso contrário
        """
        try:
            self.logger.info(f"Conectando ao servidor {self.host}:{self.port}")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)  # Timeout de 10 segundos
            
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            
            # Inicia thread para recebimento de mensagens
            self.stop_receiving.clear()
            self.receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.receive_thread.start()
            
            self.logger.info("Conectado ao servidor com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao conectar: {e}")
            return False
            
    def disconnect(self) -> None:
        """Desconecta do servidor."""
        self.logger.info("Desconectando do servidor")
        
        self.is_connected = False
        self.stop_receiving.set()
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
            
        if self.socket:
            self.socket.close()
            self.socket = None
            
        self.logger.info("Desconectado do servidor")
        
    def _receive_messages(self) -> None:
        """Thread para receber mensagens do servidor."""
        recv_buffer = b""
        while not self.stop_receiving.is_set() and self.is_connected:
            try:
                if not self.socket:
                    break

                data = self.socket.recv(4096)
                if not data:
                    self.logger.warning("Conexão fechada pelo servidor")
                    break

                recv_buffer += data

                # Processa todas as mensagens completas (terminadas com \n)
                while b"\n" in recv_buffer:
                    line, recv_buffer = recv_buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line.decode('utf-8'))
                        self._handle_server_message(message)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Erro ao decodificar mensagem: {e}")
                        continue
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_connected:
                    self.logger.error(f"Erro ao receber mensagem: {e}")
                break
                
        self.is_connected = False
        
    def _handle_server_message(self, message: Dict[str, Any]) -> None:
        """
        Processa mensagens recebidas do servidor.
        
        Args:
            message: Mensagem do servidor
        """
        message_type = message.get("type", "unknown")
        
        if message_type == "welcome":
            print(f"\n🎉 {message.get('message', 'Conectado')}")
            
        elif message_type == "recognition_result":
            self._handle_recognition_result(message)
            
        elif message_type == "image_captured":
            print(f"\n📸 Imagem capturada às {time.ctime(message.get('timestamp', time.time()))}")
            
        elif message_type == "face_added":
            print(f"\n✅ {message.get('message', 'Face adicionada')}")
            
        elif message_type == "known_faces_list":
            self._handle_faces_list(message)

        elif message_type == "model_trained":
            ok = message.get('success', False)
            print(f"\n🛠️ Treino de modelo: {'✅ OK' if ok else '❌ Falhou'}")
            faces = message.get('known_faces', [])
            dataset_counts = message.get('dataset_counts', {})
            total_images = message.get('total_images', 0)
            if faces:
                print("   👥 Pessoas no modelo:")
                for i, n in enumerate(faces, 1):
                    count = dataset_counts.get(n, 0)
                    print(f"   {i}. {n}  (📷 {count} imagens)")
            if total_images:
                print(f"   🗂️ Total de imagens no dataset: {total_images}")
            
        elif message_type == "pong":
            print(f"\n🏓 Pong recebido - Latência: {time.time() - message.get('timestamp', 0):.3f}s")
            
        elif message_type == "error":
            print(f"\n❌ Erro: {message.get('message', 'Erro desconhecido')}")

        elif message_type == "dataset_collected":
            saved = message.get('saved', 0)
            requested = message.get('requested', 0)
            name = message.get('name', '')
            print(f"\n📥 Coleta de dataset para '{name}': {saved}/{requested} imagens salvas")
            # Sinaliza para fluxos que aguardam coleta
            self._dataset_result = message
            self._dataset_event.set()

        elif message_type == "prediction_result":
            self._handle_prediction_result(message)

        elif message_type == "access_decision":
            self._handle_access_decision(message)
            
        else:
            print(f"\n📨 Mensagem recebida: {message}")
            
    def _handle_recognition_result(self, message: Dict[str, Any]) -> None:
        """Processa resultado de reconhecimento facial."""
        faces = message.get('recognized_faces', [])
        confidence_scores = message.get('confidence_scores', [])
        
        print(f"\n🔍 Resultado do Reconhecimento:")
        print(f"   📊 Faces detectadas: {len(faces)}")
        
        if faces:
            for i, (face, confidence) in enumerate(zip(faces, confidence_scores)):
                status = "✅" if face != "Desconhecido" else "❓"
                print(f"   {status} Face {i+1}: {face} (Confiança: {confidence:.2f})")
        else:
            print("   👻 Nenhuma face detectada")
            
        # Salva imagem se disponível
        image_data = message.get('image_data')
        if image_data:
            self._save_image(image_data, f"recognition_{int(time.time())}.jpg")
            
    def _handle_faces_list(self, message: Dict[str, Any]) -> None:
        """Processa lista de faces conhecidas."""
        faces = message.get('faces', [])
        count = message.get('count', 0)
        
        print(f"\n👥 Faces Conhecidas ({count}):")
        if faces:
            for i, face in enumerate(faces, 1):
                print(f"   {i}. {face}")
        else:
            print("   📭 Nenhuma face conhecida registrada")
            
    def _save_image(self, image_data: str, filename: str) -> None:
        """Salva imagem recebida do servidor."""
        try:
            # Decodifica base64
            image_bytes = base64.b64decode(image_data)
            
            # Cria diretório se não existir
            os.makedirs("captured_images", exist_ok=True)
            
            # Salva arquivo
            filepath = os.path.join("captured_images", filename)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
                
            print(f"   💾 Imagem salva: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar imagem: {e}")
            
    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Envia mensagem para o servidor.
        
        Args:
            message: Mensagem a ser enviada
            
        Returns:
            True se enviou com sucesso, False caso contrário
        """
        if not self.is_connected or not self.socket:
            print("❌ Não conectado ao servidor")
            return False
            
        try:
            data = json.dumps(message).encode('utf-8') + b"\n"
            self.socket.sendall(data)
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar mensagem: {e}")
            return False
            
    def request_face_recognition(self) -> None:
        """Solicita reconhecimento facial."""
        message = {
            "type": "recognize_face",
            "timestamp": time.time()
        }
        
        if self.send_message(message):
            print("🔍 Solicitação de reconhecimento enviada...")
        else:
            print("❌ Falha ao enviar solicitação de reconhecimento")
            
    def request_image_capture(self) -> None:
        """Solicita captura de imagem."""
        message = {
            "type": "capture_image",
            "timestamp": time.time()
        }
        
        if self.send_message(message):
            print("📸 Solicitação de captura enviada...")
        else:
            print("❌ Falha ao enviar solicitação de captura")
            
    def add_known_face_from_file(self, name: str, image_path: str) -> None:
        """Adiciona face conhecida a partir de arquivo."""
        try:
            # Lê arquivo de imagem
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                
            message = {
                "type": "add_known_face",
                "name": name,
                "image_data": image_data,
                "timestamp": time.time()
            }
            
            if self.send_message(message):
                print(f"📤 Enviando face de {name}...")
            else:
                print("❌ Falha ao enviar face")
                
        except Exception as e:
            print(f"❌ Erro ao ler arquivo {image_path}: {e}")
            
    def list_known_faces(self) -> None:
        """Lista faces conhecidas."""
        message = {
            "type": "list_known_faces",
            "timestamp": time.time()
        }
        
        if self.send_message(message):
            print("📋 Solicitando lista de faces conhecidas...")
        else:
            print("❌ Falha ao solicitar lista")
            
    def send_ping(self) -> None:
        """Envia ping para testar conectividade."""
        message = {
            "type": "ping",
            "timestamp": time.time()
        }
        
        if self.send_message(message):
            print("🏓 Ping enviado...")
        else:
            print("❌ Falha ao enviar ping")
            
    def interactive_menu(self) -> None:
        """Menu interativo para o cliente."""
        print("\n" + "="*50)
        print("🔐 CLIENTE DE RECONHECIMENTO FACIAL")
        print("="*50)
        
        while self.is_connected:
            print("\n📋 MENU DE OPÇÕES:")
            print("1. 🔍 Reconhecer face")
            print("2. ➕ Adicionar face conhecida")
            print("3. 👥 Listar faces conhecidas")
            print("4. 🏓 Ping")
            print("5. �️ Treinar modelo (LBPH)")
            print("6. 🤖 Reconhecer e identificar (LBPH)")
            print("7. 🧹 Limpar modelo")
            print("8. 🔐 Autorizar acesso (votação 2/3)")
            print("9. � Sair")
            
            try:
                choice = input("\n👆 Escolha uma opção (1-9): ").strip()
                
                if choice == "1":
                    self.request_face_recognition()
                    
                elif choice == "2":
                    name = input("👤 Nome da pessoa: ").strip()
                    if name:
                        mode = input("📷 Deseja tirar as fotos agora (guiado) [G] ou informar um caminho com fotos [C]? (G/C): ").strip().lower()
                        if mode == 'g' or mode == '':
                            self._guided_dataset_capture(name)
                        else:
                            image_path = input("📁 Informe o caminho (arquivo ou pasta com fotos): ").strip()
                            if not os.path.exists(image_path):
                                print("❌ Caminho não encontrado")
                            elif os.path.isdir(image_path):
                                self.add_known_faces_from_dir(name, image_path)
                            else:
                                # Arquivo único
                                self.add_known_face_from_file(name, image_path)
                    else:
                        print("❌ Nome não pode estar vazio")
                        
                elif choice == "3":
                    self.list_known_faces()
                    
                elif choice == "4":
                    self.send_ping()
                    
                elif choice == "5":
                    self.send_message({"type": "train_model", "timestamp": time.time()})
                    print("🛠️ Treinando modelo...")

                elif choice == "6":
                    print("🧭 Posicione o rosto na frente da câmera e mantenha-se estável por um instante...")
                    self.send_message({"type": "predict", "timestamp": time.time()})
                    print("🤖 Solicitando identificação (LBPH)...")

                elif choice == "7":
                    self.send_message({"type": "clear_model", "timestamp": time.time()})
                    print("🧹 Limpando modelo/dataset...")

                elif choice == "8":
                    print("🔐 Autorização por votação: por padrão, 3 frames e 2 acertos (abaixo do limiar) → PERMITIR.")
                    try:
                        # Parâmetros opcionais
                        raw_count = input(f"   ➕ Quantidade de frames [3]: ").strip()
                        count = int(raw_count) if raw_count else 3
                        if count <= 0:
                            print("   ⚠️ Valor inválido, usando 3.")
                            count = 3

                        default_required = 2 if count >= 2 else 1
                        raw_req = input(f"   🗳️ Votos necessários [{default_required}]: ").strip()
                        required = int(raw_req) if raw_req else default_required
                        if required <= 0 or required > count:
                            print(f"   ⚠️ Votos necessários devem estar entre 1 e {count}. Ajustando para {default_required}.")
                            required = default_required

                        raw_thr = input(f"   🎚️ Limiar LBPH [{LBPH_THRESHOLD}]: ").strip()
                        threshold = float(raw_thr) if raw_thr else LBPH_THRESHOLD
                    except Exception:
                        print("   ⚠️ Entrada inválida; usando parâmetros padrão (3/2, limiar do config).")
                        count, required, threshold = 3, 2, LBPH_THRESHOLD

                    self.request_authorize_access(count=count, required=required, threshold=threshold)

                elif choice == "9":
                    print("👋 Encerrando cliente...")
                    break
                    
                else:
                    print("❌ Opção inválida")
                    
                # Pausa breve para mostrar resultado
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\n👋 Encerrando cliente...")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")

    def _guided_dataset_capture(self, name: str) -> None:
        """Fluxo guiado: 6 passos, 3 fotos por passo (total 18)."""
        total_saved = 0
        steps = [
            "Olhe de frente para a câmera",
            "Vire levemente o rosto para a esquerda",
            "Vire levemente o rosto para a direita",
            "Incline levemente o rosto para cima",
            "Incline levemente o rosto para baixo",
            "Faça uma expressão diferente (sorria ou relaxe)"
        ]
        print(f"\n🎬 Iniciando coleta guiada para '{name}' (6 passos, 3 fotos por passo)\n")
        for idx, instr in enumerate(steps, start=1):
            print(f"Passo {idx}/6: {instr}")
            input("   ➡️ Posicione-se e pressione Enter para capturar 3 fotos...")
            # Envia solicitação para capturar 3 imagens
            self._dataset_event.clear()
            self._dataset_result = None
            ok = self.send_message({
                "type": "collect_dataset",
                "name": name,
                "count": 3,
                "timestamp": time.time()
            })
            if not ok:
                print("❌ Falha ao enviar solicitação de coleta")
                continue
            print("   📸 Capturando 3 fotos...")
            # Aguarda confirmação do servidor
            signaled = self._dataset_event.wait(timeout=15.0)
            if signaled and self._dataset_result:
                saved = int(self._dataset_result.get('saved', 0))
                total_saved += saved
                print(f"   ✅ Passo {idx} concluído: {saved}/3 fotos salvas")
            else:
                print(f"   ⚠️ Passo {idx}: sem confirmação do servidor (tente melhorar a iluminação/posição)")
        print(f"\n📦 Coleta finalizada: {total_saved}/18 fotos salvas para '{name}'.")

    def _handle_prediction_result(self, message: Dict[str, Any]) -> None:
        """Apresenta o resultado da predição de forma amigável e focada em controle de acesso."""
        faces = message.get('recognized_faces', [])
        confidence_scores = message.get('confidence_scores', [])
        print("\n🔐 Identificação (LBPH):")
        if not faces:
            print("   👻 Nenhuma face detectada")
        else:
            for i, name in enumerate(faces):
                conf = None
                if i < len(confidence_scores):
                    try:
                        conf = float(confidence_scores[i])
                    except Exception:
                        conf = None
                known = name and name != "Desconhecido"
                if known and conf is not None and conf <= LBPH_THRESHOLD:
                    verdict = "✅ Acesso permitido"
                    detail = f"(confiança: {conf:.2f} ≤ limite {LBPH_THRESHOLD})"
                elif known and conf is not None:
                    verdict = "🚫 Acesso negado"
                    detail = f"(confiança: {conf:.2f} > limite {LBPH_THRESHOLD})"
                else:
                    verdict = "🚫 Acesso negado"
                    detail = "(não reconhecido)"
                label = name if known else "Desconhecido"
                print(f"   👤 Face {i+1}: {label} {detail} → {verdict}")
            if any((f and f != "Desconhecido") for f in faces):
                print("   ℹ️ Dica: ajuste o LBPH_THRESHOLD no config conforme necessário.")
            else:
                print("   💡 Dica: colete mais imagens variadas e treine o modelo (opções 2 e 5).")

        image_data = message.get('image_data')
        if image_data:
            self._save_image(image_data, f"predict_{int(time.time())}.jpg")

    def request_authorize_access(self, count: int = 3, required: int = 2, threshold: float = None) -> None:
        """Solicita decisão de acesso com votação majoritária."""
        if threshold is None:
            threshold = LBPH_THRESHOLD
        msg = {
            "type": "authorize_access",
            "count": count,
            "required": required,
            "threshold": threshold,
            "timestamp": time.time(),
        }
        if self.send_message(msg):
            print(f"🔎 Verificando acesso com votação {required}/{count} (limite {threshold})...")
        else:
            print("❌ Falha ao enviar solicitação de autorização")

    def _handle_access_decision(self, message: Dict[str, Any]) -> None:
        """Exibe a decisão de acesso e salva um snapshot."""
        granted = message.get('granted', False)
        name = message.get('name') or 'Desconhecido'
        votes = message.get('votes', 0)
        required = message.get('required', 2)
        count = message.get('count', 3)
        threshold = message.get('threshold', LBPH_THRESHOLD)
        print("\n🗳️ Resultado da votação de acesso:")
        print(f"   👤 Usuário: {name}")
        print(f"   🗳️ Votos: {votes}/{count} (mínimo {required}) | Limite: {threshold}")
        print("   ✅ Acesso PERMITIDO" if granted else "   🚫 Acesso NEGADO")
        if not granted:
            print("   💡 Dica: melhore iluminação/posição, aumente dataset e ajuste o limite no config.")
        img = message.get('image_data')
        if img:
            self._save_image(img, f"access_{int(time.time())}.jpg")

    def add_known_faces_from_dir(self, name: str, dir_path: str) -> None:
        """Envia todas as imagens de um diretório como faces conhecidas para a mesma pessoa."""
        try:
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts]
            if not files:
                print("📂 Pasta não contém imagens suportadas (.jpg, .jpeg, .png, .bmp)")
                return
            print(f"📤 Enviando {len(files)} imagens de '{dir_path}' para '{name}'...")
            sent = 0
            for f in files:
                full = os.path.join(dir_path, f)
                self.add_known_face_from_file(name, full)
                sent += 1
                # pequeno intervalo para não saturar o socket
                time.sleep(0.05)
            print(f"✅ Envio concluído: {sent} imagens enviadas para '{name}'.")
        except Exception as e:
            print(f"❌ Erro ao enviar pasta: {e}")


def main():
    """Função principal do cliente."""
    parser = argparse.ArgumentParser(description="Cliente do servidor de reconhecimento facial")
    parser.add_argument("--host", default=SERVER_HOST, help=f"Endereço do servidor (default: {SERVER_HOST})")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help=f"Porta do servidor (default: {SERVER_PORT})")
    args = parser.parse_args()

    client = FacialRecognitionClient(host=args.host, port=args.port)
    
    try:
        # Conecta ao servidor
        if client.connect():
            # Inicia menu interativo
            client.interactive_menu()
        else:
            print("❌ Falha ao conectar ao servidor")
            
    except KeyboardInterrupt:
        print("\n👋 Cliente interrompido pelo usuário")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
