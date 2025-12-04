# 🔐 Sistema de Reconhecimento Facial para Raspberry Pi 3

Sistema de controle de acesso por reconhecimento facial otimizado para **Raspberry Pi 3**.  
Utiliza **OpenCV LBPH** (Local Binary Patterns Histograms) - algoritmo leve e eficiente para hardware com recursos limitados.

## 📋 Características

- ✅ **Leve**: Funciona no Raspberry Pi 3 (1GB RAM)
- ✅ **Sem TensorFlow/DeepFace**: Usa apenas OpenCV
- ✅ **Arquitetura Cliente-Servidor**: API REST com Flask
- ✅ **Autenticação JWT**: Endpoints administrativos protegidos
- ✅ **Multi-worker**: Suporte a conexões simultâneas via Gunicorn + ThreadPool
- ✅ **Interface Web**: Dashboard administrativo completo
- ✅ **Detecção de Intrusão**: Bloqueio automático de IPs suspeitos
- ✅ **Backup Automático**: Sistema de backup agendado e manual

## 🆕 Novas Funcionalidades (v2.0)

### 🌐 Interface Web Administrativa
Acesse `http://IP_DO_SERVIDOR:5000/web/` para:
- **Dashboard**: Estatísticas em tempo real
- **Usuários**: Cadastro e gerenciamento de faces
- **Segurança**: Monitoramento de IPs bloqueados e alertas
- **Backups**: Criar, restaurar e gerenciar backups
- **Logs**: Visualização de acessos

### 🔒 Detecção de Intrusão
- Bloqueio automático após 5 tentativas falhas
- Tempo de bloqueio configurável (padrão: 15 min)
- Alertas de segurança em tempo real
- Desbloqueio manual via interface web

### 💾 Sistema de Backup
- Backup automático a cada 24 horas
- Backup manual via CLI ou interface web
- Restauração com um clique
- Limite de 10 backups (rotação automática)

### ⚡ ThreadPool para Alta Performance
- Processamento paralelo de requisições
- Configurável via variável de ambiente `MAX_WORKERS`

## 🏗️ Arquitetura

```
┌─────────────────┐     HTTP/REST     ┌─────────────────────────┐
│ Cliente Admin   │◄─────────────────►│   Servidor Flask        │
│ (Notebook)      │                   │   (Raspberry Pi 3)      │
└─────────────────┘                   │                         │
                                      │  • OpenCV LBPH          │
┌─────────────────┐     HTTP/REST     │  • SQLite               │
│ Cliente Kiosk   │◄─────────────────►│  • Gunicorn + ThreadPool│
│ (Terminal)      │                   │  • Backup System        │
└─────────────────┘                   │  • Intrusion Detection  │
        │                             └─────────────────────────┘
        │                                        │
┌─────────────────┐                   ┌─────────────────────────┐
│ Browser Web     │◄─────────────────►│   Interface Web Admin   │
│ (Dashboard)     │                   │   /web/dashboard        │
└─────────────────┘                   └─────────────────────────┘
```

## 📁 Estrutura do Projeto

```
topicosEspeciais/
├── servidor_pi/              # Servidor (roda no Raspberry Pi)
│   ├── app.py               # Aplicação Flask + endpoints
│   ├── auth_utils.py        # Autenticação bcrypt
│   ├── database.py          # Operações SQLite
│   ├── face_utils.py        # Reconhecimento facial LBPH
│   ├── security.py          # Detecção de intrusão
│   ├── backup.py            # Sistema de backup
│   ├── templates/           # Templates HTML da interface web
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── users.html
│   │   ├── security.html
│   │   ├── backups.html
│   │   └── logs.html
│   ├── requirements.txt     # Dependências do servidor
│   └── setup_raspberry.sh   # Script de instalação
│
├── cliente_admin/            # Cliente de gerenciamento
│   ├── admin.py             # Interface de administração
│   └── requirements.txt
│
├── cliente_acesso/           # Cliente de verificação
│   ├── kiosk.py             # Interface do kiosk
│   └── requirements.txt
│
├── models/                   # Modelo LBPH treinado (gerado automaticamente)
├── backups/                  # Backups do sistema (gerado automaticamente)
└── readme.txt               # Instruções detalhadas de instalação
```

## 🚀 Instalação Rápida

### No Raspberry Pi (Servidor)

```bash
cd servidor_pi
chmod +x setup_raspberry.sh
./setup_raspberry.sh
```

Depois, crie um admin e inicie o servidor:

```bash
source venv/bin/activate
export FLASK_APP=servidor_pi.app
flask add-admin admin senha123
gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 "servidor_pi.app:app"
```

### No PC/Notebook (Clientes)

**Cliente Admin:**
```bash
cd cliente_admin
pip install -r requirements.txt

# Configure o IP do servidor
# PowerShell:
$env:SERVER_URL="http://IP_DO_RASPBERRY:5000"

# Linux/Mac:
export SERVER_URL="http://IP_DO_RASPBERRY:5000"

python admin.py
```

**Cliente Kiosk:**
```bash
cd cliente_acesso
pip install -r requirements.txt

# Configure o IP do servidor (mesmo procedimento acima)
python kiosk.py
```

### Interface Web
Abra no navegador: `http://IP_DO_RASPBERRY:5000/web/`

## 🔌 API REST

### Endpoints Públicos

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Verifica status do servidor |
| POST | `/verify` | Verifica face e libera/nega acesso |

### Endpoints Administrativos (JWT)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/admin/login` | Autentica e retorna token JWT |
| POST | `/admin/users/add` | Cadastra usuário com face |
| GET | `/admin/users` | Lista todos os usuários |
| DELETE | `/admin/users/{id}` | Remove um usuário |
| GET | `/admin/model/stats` | Estatísticas do modelo LBPH |
| GET | `/admin/security/stats` | Estatísticas de segurança |
| POST | `/admin/backup/create` | Cria backup manual |
| GET | `/admin/backup/list` | Lista backups disponíveis |

### Interface Web

| Rota | Descrição |
|------|-----------|
| `/web/login` | Login da interface web |
| `/web/dashboard` | Dashboard principal |
| `/web/users` | Gerenciamento de usuários |
| `/web/security` | Monitor de segurança |
| `/web/backups` | Gerenciamento de backups |
| `/web/logs` | Logs de acesso |

## 📖 Documentação Completa

Consulte o arquivo `readme.txt` para:
- Instalação passo a passo detalhada
- Configuração de produção
- Solução de problemas
- Ajuste de limiares LBPH

## 🔧 Tecnologias Utilizadas

| Componente | Tecnologia |
|------------|------------|
| Servidor Web | Flask + Gunicorn |
| ThreadPool | concurrent.futures |
| Autenticação | JWT (Flask-JWT-Extended) |
| Banco de Dados | SQLite |
| Reconhecimento Facial | OpenCV LBPH |
| Hash de Senhas | bcrypt |
| Interface Web | Jinja2 + Bootstrap |

## ⚙️ Configurações

### Variáveis de Ambiente

```bash
# Chave JWT (OBRIGATÓRIO em produção)
export JWT_SECRET_KEY="sua-chave-secreta-super-segura"

# Chave de sessão (OBRIGATÓRIO em produção)
export SECRET_KEY="chave-de-sessao-segura"

# Workers do ThreadPool (padrão: 4)
export MAX_WORKERS=4

# IP do servidor para clientes
export SERVER_URL="http://192.168.1.100:5000"
```

### Limiar LBPH

O limiar de reconhecimento pode ser ajustado em `servidor_pi/face_utils.py`:

```python
LBPH_THRESHOLD = 80.0  # Menor = mais restritivo
```

### Configuração de Segurança

Em `servidor_pi/security.py`:

```python
MAX_FAILED_ATTEMPTS = 5      # Tentativas antes de bloquear
BLOCK_DURATION_MINUTES = 15  # Duração do bloqueio
```

### Workers do Gunicorn

Para Raspberry Pi 3 (1GB RAM), use no máximo 2 workers:

```bash
gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 "servidor_pi.app:app"
```

## 📊 Comandos CLI

```bash
# Inicializar banco de dados
flask init-db

# Adicionar administrador
flask add-admin <usuario> <senha>

# Ver estatísticas do modelo
flask model-stats

# Limpar modelo LBPH
flask clear-model

# Criar backup manual
flask backup

# Iniciar backup automático
flask start-auto-backup

# Rodar Cliente Kiosk
$env:SERVER_URL="http://192.168.1.30:5000"; pip install -r requirements.txt; python kiosk.py
#ou
$env:SERVER_URL="http://192.168.1.30:5000"; python kiosk.py

# Rodar Cliente Admin
$env:SERVER_URL="http://192.168.1.30:5000"; pip install -r requirements.txt; python admin.py
#ou
$env:SERVER_URL="http://192.168.1.30:5000"; python admin.py

```

## 📞 Contato
* Desenvolvedor: Nalbert Schwank Costa Santos, João Paulo Bastos Sampaio, Luiz Fernando Brito Ferreira, Lucas Resende, Carlos Castro 
* Projeto: Tópicos Especiais em Computação
* Objetivo: Sistema de Controle de Acesso com Reconhecimento Facial

## 📄 Licença

MIT License
