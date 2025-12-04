# Sistema de Reconhecimento Facial - Raspberry Pi 3

Sistema de controle de acesso por reconhecimento facial otimizado para Raspberry Pi 3.
Usa OpenCV LBPH (Local Binary Patterns Histograms) - leve e eficiente para hardware limitado.

## Arquitetura

```
┌─────────────────┐     HTTP/REST     ┌─────────────────────────┐
│ Cliente Admin   │◄─────────────────►│   Servidor Flask        │
│ (Notebook)      │                   │   (Raspberry Pi 3)      │
└─────────────────┘                   │                         │
                                      │  - OpenCV LBPH          │
┌─────────────────┐     HTTP/REST     │  - SQLite               │
│ Cliente Kiosk   │◄─────────────────►│  - Gunicorn (produção)  │
│ (Terminal)      │                   └─────────────────────────┘
└─────────────────┘
```

## Requisitos do Sistema

### Raspberry Pi 3 (Servidor)
- Raspberry Pi OS (32-bit ou 64-bit)
- Python 3.9+
- Câmera não necessária no servidor (recebe imagens via HTTP)

### Clientes (Notebook/PC)
- Windows, Linux ou macOS
- Python 3.9+
- Webcam

---

## Instalação no Raspberry Pi 3 (Servidor)

### 1. Atualizar o sistema
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Instalar dependências do sistema
```bash
sudo apt install -y python3-pip python3-venv libatlas-base-dev
```

### 3. Criar ambiente virtual
```bash
cd servidor_pi
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar dependências Python
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Nota: A instalação do opencv-contrib-python-headless pode demorar alguns minutos no Pi.

### 5. Inicializar o banco de dados
```bash
export FLASK_APP=servidor_pi.app
flask init-db
```

### 6. Criar o primeiro administrador
```bash
flask add-admin admin senha123
```

### 7. Iniciar o servidor (Desenvolvimento)
```bash
python -m servidor_pi.app
```

### 8. Iniciar o servidor (Produção com Gunicorn)
```bash
gunicorn --workers 2 --bind 0.0.0.0:5000 "servidor_pi.app:app"
```

Nota: Use 2 workers no Pi 3 (4 pode sobrecarregar a RAM de 1GB).

---

## Instalação nos Clientes (Notebook/PC)

### Cliente Admin (Gerenciamento)

```bash
cd cliente_admin
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Antes de executar, configure o IP do servidor:
```bash
# Windows PowerShell
$env:SERVER_URL="http://IP_DO_RASPBERRY:5000"

# Linux/Mac
export SERVER_URL="http://IP_DO_RASPBERRY:5000"
```

Executar:
```bash
python admin.py
```

### Cliente Kiosk (Verificação de Acesso)

```bash
cd cliente_acesso
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Configurar IP do servidor:
```bash
# Windows PowerShell
$env:SERVER_URL="http://IP_DO_RASPBERRY:5000"

# Linux/Mac
export SERVER_URL="http://IP_DO_RASPBERRY:5000"
```

Executar:
```bash
python kiosk.py
```

---

## Uso do Sistema

### 1. Cadastrar Usuários (Cliente Admin)
1. Execute o cliente admin
2. Faça login com as credenciais do administrador
3. Selecione "Adicionar Usuário"
4. Digite o nome e capture a foto do rosto

### 2. Verificar Acesso (Cliente Kiosk)
1. Execute o cliente kiosk
2. Pressione ESPAÇO para verificar a face
3. O sistema mostrará se o acesso foi liberado ou negado

---

## API REST

### Endpoints Públicos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | /health | Verifica status do servidor |
| POST | /verify | Verifica face e retorna se acesso é liberado |

### Endpoints Autenticados (JWT)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | /admin/login | Autentica admin e retorna token |
| POST | /admin/users/add | Adiciona novo usuário com face |
| GET | /admin/users | Lista todos os usuários |
| DELETE | /admin/users/{id} | Remove usuário |
| GET | /admin/model/stats | Estatísticas do modelo LBPH |

---

## Comandos Flask CLI

```bash
# Inicializar banco de dados
flask init-db

# Adicionar administrador
flask add-admin <username> <password>

# Ver estatísticas do modelo
flask model-stats

# Limpar modelo LBPH
flask clear-model
```

---

## Solução de Problemas

### Erro: "Nenhum rosto detectado"
- Melhore a iluminação
- Posicione o rosto de frente para a câmera
- Mantenha distância adequada (30-80cm)

### Erro de conexão com servidor
- Verifique se o servidor está rodando
- Confirme o IP do Raspberry Pi: `hostname -I`
- Verifique firewall: `sudo ufw allow 5000`

### Modelo não reconhece usuários
- Cadastre mais fotos do mesmo usuário (diferentes ângulos)
- Ajuste o limiar em `face_utils.py` (LBPH_THRESHOLD)
- Valores menores = mais restritivo

### Servidor lento ou travando
- Use apenas 2 workers no Gunicorn
- Verifique uso de memória: `free -h`
- Considere usar swap: `sudo dphys-swapfile setup`

---

## Estrutura de Arquivos

```
topicosEspeciais/
├── servidor_pi/          # Servidor (roda no Raspberry Pi)
│   ├── app.py           # Aplicação Flask
│   ├── auth_utils.py    # Funções de autenticação
│   ├── database.py      # Operações SQLite
│   ├── face_utils.py    # Reconhecimento LBPH
│   └── requirements.txt
│
├── cliente_admin/        # Cliente de gerenciamento
│   ├── admin.py         # Interface de administração
│   └── requirements.txt
│
├── cliente_acesso/       # Cliente de verificação
│   ├── kiosk.py         # Interface do kiosk
│   └── requirements.txt
│
├── models/              # Armazena modelo LBPH treinado
└── logs/                # Logs do sistema
```

---

## Segurança

- **JWT_SECRET_KEY**: Altere a chave em `app.py` para produção
- Use HTTPS em produção (configure nginx como proxy reverso)
- Não exponha a porta 5000 diretamente à internet

---

## Licença

MIT License