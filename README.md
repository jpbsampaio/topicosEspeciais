## Sistema de Reconhecimento Facial (Versão Enxuta)

Projeto reduzido para o essencial: servidor + cliente TCP, captura de câmera e reconhecimento facial usando somente OpenCV (LBPH).
### Estrutura Mantida
```
src/
    server.py                # Servidor TCP (JSON por linha) + roteamento
    client.py                # Cliente interativo (menu)
    camera_handler.py        # Abstração da câmera (captura e encode JPEG)
    face_recognition_handler_compatible.py  # Handler LBPH (OpenCV)
    config.py                # Configurações centrais
models/                    # Modelo LBPH e labels
data/                      # Dataset organizado por pessoa (data/<nome>/*.jpg)
logs/                      # Logs do servidor/cliente
captured_images/           # Snapshots salvos pelo cliente (predict / autorização)
### Dependências
Instale apenas:
```
pip install -r requirements.txt
```

### Uso Rápido
1. Inicie o servidor:
```
python src/server.py
2. Em outro terminal, inicie o cliente:
```
python src/client.py
3. No cliente:
    - Opção 2: adicionar face (coleta guiada ou pasta)
    - Opção 5: treinar modelo (gera/atualiza `models/opencv_lbph.xml`)
    - Opção 6: identificar (usa LBPH)
    - Opção 8: votação de autorização (N frames, K votos)

### Dataset
Imagens ficam em `data/<nome>/*.jpg` (tons de cinza recortados). Quanto mais variedade (ângulos/iluminação), melhor.

### Limiar LBPH
O valor usado para aceitar predição é configurado em `config.py` (LBPH_THRESHOLD). Distâncias menores indicam melhor correspondência.

### Limpeza Realizada
Removidos:
- Handlers redundantes (`face_recognition_handler.py`, `opencv_face_handler.py`)
- Scripts de exemplo e testes (`demo.py`, `quick_test.py`, `setup_example.py`)
- Arquivos de status/documentação intermediária (`STATUS.md`)

### Próximos Passos (Opcional)
- Adicionar `.env` para configurar host/porta/câmera sem editar código.
- Criar testes unitários mínimos para o handler LBPH.
- Adicionar persistência de métricas ou API REST (futuro).

---
Projeto minimalista pronto para extensão ou deploy.
# 🔐 Sistema de Reconhecimento Facial com Arquitetura Cliente-Servidor

Sistema avançado de reconhecimento facial desenvolvido em Python com arquitetura cliente-servidor usando sockets TCP e ThreadPool para gerenciamento eficiente de múltiplas conexões simultâneas.

## 🎯 Objetivo

## (Seção removida – documentação antiga substituída pela versão enxuta no início do arquivo)

```bash
cd src
python server.py
```

O servidor será iniciado em `localhost:8888` com as seguintes características:
- **ThreadPool**: 5 workers por padrão
- **Logging**: Logs salvos em `server.log`
- **Conexões Simultâneas**: Suporte para múltiplos clientes

### 👤 Executar o Cliente

```bash
cd src
python client.py
```

O cliente oferece um menu interativo com as seguintes opções:
1. **🔍 Reconhecer Face**: Captura e analisa faces do frame atual
2. **➕ Adicionar Face Conhecida**: Coleta guiada (6 passos × 3 fotos) ou importa de pasta/arquivo
3. **👥 Listar Faces Conhecidas**: Mostra todas as pessoas cadastradas
4. **🏓 Ping**: Testa conectividade com o servidor
5. **🛠️ Treinar modelo (LBPH)**: Re-treina com as imagens em `data/<nome>/`
6. **🤖 Reconhecer e identificar (LBPH)**: Predição com limiar configurável
7. **🧹 Limpar modelo**: Limpa dataset/modelos
8. **🔐 Autorizar acesso (votação)**: Janela de votação com parâmetros configuráveis
9. **🚪 Sair**

Notas:
- A opção “Capturar Imagem” foi removida em favor do fluxo de coleta guiada ou importação por pasta.
- As imagens coletadas ficam em `data/<nome>/` e os modelos em `models/`.

### 🧭 Coleta Guiada de Dataset

- 6 passos com instruções (frente, esquerda, direita, cima, baixo, expressão)
- 3 fotos por passo (total 18), salvas em `data/<nome>/`
- Alternativamente, importe fotos de um diretório com imagens do rosto

### 🤖 Treino e Predição (LBPH)

- Após coletar dados, use “Treinar modelo (LBPH)”
- A predição utiliza um limiar (`LBPH_THRESHOLD`) para decidir se um rosto conhecido é aceito
- As imagens de predição são salvas em `captured_images/`

### 🔐 Autorização por Votação (2/3 por padrão)

- Captura N frames (padrão 3) e exige R acertos (padrão 2) abaixo do limiar para permitir
- Parâmetros configuráveis no cliente: quantidade de frames, votos necessários, limiar
- Útil para reduzir falsos positivos em ambientes variáveis

## 🔧 Funcionalidades Técnicas

### 🔌 Arquitetura Cliente-Servidor

- **Protocolo**: TCP Sockets
- **Formato de Dados**: JSON
- **Encoding**: UTF-8
- **Threading**: ThreadPoolExecutor para conexões simultâneas

### 🧠 Reconhecimento Facial

- Modo compatível com OpenCV (LBPH) ativado por padrão
- Se `face_recognition` estiver instalado, o handler original pode ser usado
- Limiar de decisão LBPH configurável via `LBPH_THRESHOLD`

### 📹 Gerenciamento de Câmera

- **Captura**: OpenCV VideoCapture
- **Threading**: Captura contínua em thread separada
- **Buffer**: Queue para frames com controle de latência
- **Formatos**: Suporte para JPEG e PNG

### 🔄 Gerenciamento de Conexões

```python
# Exemplo de uso do ThreadPool no servidor
self.executor = ThreadPoolExecutor(max_workers=5)
future = self.executor.submit(self.handle_client, client_socket, client_address)
```

## 📋 Protocolo de Comunicação

### 📤 Mensagens do Cliente para Servidor

```json
{
    "type": "recognize_face",
    "timestamp": 1234567890.123
}
```

```json
{
    "type": "add_known_face",
    "name": "João Silva",
    "image_data": "base64_encoded_image",
    "timestamp": 1234567890.123
}
```

### 📥 Respostas do Servidor

```json
{
    "type": "recognition_result",
    "recognized_faces": ["João Silva", "Desconhecido"],
    "confidence_scores": [0.95, 0.0],
    "image_data": "base64_encoded_image",
    "timestamp": 1234567890.123
}
```

## 🍓 Configuração para Raspberry Pi

### 1. 🔧 Habilitar Câmera

```bash
sudo raspi-config
# Interface Options -> Camera -> Enable
```

### 2. 📦 Instalar Dependências

```bash
sudo apt update
sudo apt install python3-opencv python3-pip cmake build-essential
pip3 install -r requirements.txt
```

### 3. 🔌 Configuração GPIO (Futuro - Fechadura)

```python
import RPi.GPIO as GPIO

# Pino para controle da fechadura
LOCK_PIN = 18

def unlock_door():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LOCK_PIN, GPIO.OUT)
    GPIO.output(LOCK_PIN, GPIO.HIGH)
    time.sleep(2)  # Mantém desbloqueado por 2 segundos
    GPIO.output(LOCK_PIN, GPIO.LOW)
    GPIO.cleanup()
```

## 🔧 Configurações Avançadas

### ⚙️ Parâmetros do Servidor

```python
server = FacialRecognitionServer(
    host='0.0.0.0',        # Aceita conexões de qualquer IP
    port=8888,             # Porta do servidor
    max_workers=10         # Número máximo de threads
)
```

### ⚙️ Configuração via .env (opcional)

Crie um arquivo `.env` na raiz com variáveis (todas possuem defaults):

```
SERVER_HOST=localhost
SERVER_PORT=8888
MAX_WORKERS=5

# Câmera
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Pastas
MODELS_DIR=models
DATA_DIR=data
LOG_DIR=logs

# LBPH
LBPH_THRESHOLD=65.0
```

Observações:
- LBPH_THRESHOLD menor → mais restritivo (menos falsos positivos, mais falsos negativos)
- Ajuste conforme iluminação/qualidade das imagens

### ⚙️ Parâmetros da Câmera

```python
camera = CameraHandler(
    camera_index=0,                # Índice da câmera
    resolution=(640, 480)          # Resolução da captura
)
```

## 📊 Monitoramento e Logs

O sistema gera logs detalhados em:
- `server.log` - Logs do servidor
- `client.log` - Logs do cliente
- `logs/system.log` - Logs gerais do sistema

### 📈 Estatísticas do Servidor

```python
stats = server.get_server_stats()
print(f"Conexões ativas: {stats['active_connections']}")
print(f"Clientes conectados: {stats['connected_clients']}")
```

## 🧪 Testes e Debugging

### 🔍 Executar Testes Completos

```bash
python setup_example.py
# Escolha opção 1 para testes completos
```

### 🐛 Debug Mode

Para ativar modo debug, modifique o nível de logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Roadmap Futuro

- [ ] 🔐 Integração com fechadura eletrônica
- [ ] 🌐 Interface web para administração
- [ ] 📱 Aplicativo móvel para controle
- [ ] 🔄 Backup automático de dados
- [ ] 📧 Notificações por email/SMS
- [ ] 🎯 Detecção de tentativas de invasão
- [ ] 📊 Dashboard de estatísticas
- [ ] 🔒 Criptografia de comunicação

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🆘 Suporte

Se você encontrar problemas:

1. **🔍 Verifique os logs** em `logs/` e `*.log`
2. **🧪 Execute os testes** com `python setup_example.py`
3. **📖 Consulte a documentação** neste README
4. **🐛 Abra uma issue** descrevendo o problema

## 📞 Contato

- **Desenvolvedor**: João Pedro
- **Projeto**: Tópicos Especiais em Computação
- **Objetivo**: Sistema de Controle de Acesso com Reconhecimento Facial

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!** ⭐
