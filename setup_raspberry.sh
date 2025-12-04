#!/bin/bash
# Script de instalação do servidor no Raspberry Pi 3
# Execute da RAIZ do projeto: chmod +x setup_raspberry.sh && ./setup_raspberry.sh

set -e

echo "========================================"
echo "  Sistema de Reconhecimento Facial v2.0"
echo "  para Raspberry Pi 3"
echo "========================================"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verifica se está na raiz do projeto
if [ ! -d "servidor_pi" ]; then
    echo -e "${RED}Erro: Execute este script da raiz do projeto (onde está a pasta servidor_pi)${NC}"
    exit 1
fi

# Verifica se está rodando no Pi
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}Aviso: Este script foi projetado para Raspberry Pi${NC}"
fi

echo ""
echo -e "${GREEN}[1/8] Atualizando o sistema...${NC}"
sudo apt update

echo ""
echo -e "${GREEN}[1.5/8] Verificando espaço em disco...${NC}"
FREE_SPACE=$(df / | tail -1 | awk '{print $4}')
if [ "$FREE_SPACE" -lt 1000000 ]; then
    echo -e "${RED}AVISO: Espaço em disco baixo! (< 1GB livre)${NC}"
    echo "Limpando caches para liberar espaço..."
    sudo apt clean
    sudo apt autoremove -y
    pip cache purge 2>/dev/null || true
    rm -rf ~/.cache/pip 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}[2/8] Instalando dependências do sistema...${NC}"
# libopenblas-dev substitui libatlas-base-dev em versões mais novas do Raspberry Pi OS
# python3-opencv instala OpenCV pré-compilado (MUITO mais rápido que pip)
sudo apt install -y python3-pip python3-venv libopenblas-dev libjpeg-dev libpng-dev libtiff-dev python3-opencv

echo ""
echo -e "${GREEN}[3/8] Criando ambiente virtual...${NC}"
if [ -d "venv" ]; then
    echo "Ambiente virtual já existe, pulando..."
else
    # --system-site-packages permite usar o OpenCV instalado via apt
    python3 -m venv --system-site-packages venv
fi
source venv/bin/activate

echo ""
echo -e "${GREEN}[4/8] Atualizando pip...${NC}"
pip install --upgrade pip

echo ""
echo -e "${GREEN}[5/8] Instalando dependências Python...${NC}"
echo "OpenCV já foi instalado via apt (muito mais rápido!)"

# Limpa cache do pip para economizar espaço
pip cache purge 2>/dev/null || true

# Verifica se OpenCV está funcionando
python3 -c "import cv2; print(f'OpenCV versão: {cv2.__version__}')" || {
    echo -e "${RED}Erro: OpenCV não encontrado. Tente: sudo apt install python3-opencv${NC}"
    exit 1
}

# Instala o resto das dependências (sem OpenCV, já instalado)
pip install --no-cache-dir -r servidor_pi/requirements.txt

echo ""
echo -e "${GREEN}[6/8] Criando diretórios necessários...${NC}"
mkdir -p models
mkdir -p backups
mkdir -p logs

echo ""
echo -e "${GREEN}[7/8] Configurando variáveis de ambiente...${NC}"
if [ ! -f ".env" ]; then
    if [ -f "servidor_pi/.env.example" ]; then
        cp servidor_pi/.env.example .env
        # Gera chaves secretas aleatórias
        JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        SESSION_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        sed -i "s/sua-chave-jwt-super-secreta-mude-isso/$JWT_SECRET/g" .env
        sed -i "s/sua-chave-sessao-super-secreta-mude-isso/$SESSION_SECRET/g" .env
        echo -e "${GREEN}Arquivo .env criado com chaves seguras!${NC}"
    else
        echo -e "${YELLOW}Arquivo .env.example não encontrado. Criando .env básico...${NC}"
        cat > .env << EOF
# Configurações do Servidor
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
MAX_WORKERS=4
FLASK_APP=servidor_pi.app
FLASK_ENV=production
EOF
    fi
else
    echo "Arquivo .env já existe, mantendo configurações atuais."
fi

echo ""
echo -e "${GREEN}[8/8] Inicializando banco de dados...${NC}"
# Inicializa o banco de dados usando o módulo do projeto
python -c "import servidor_pi.database as db; db.init_db()"

echo ""
echo "========================================"
echo -e "${GREEN}  Instalação concluída!${NC}"
echo "========================================"
echo ""
echo -e "${BLUE}Próximos passos:${NC}"
echo ""
echo "1. Ative o ambiente e carregue variáveis:"
echo "   source venv/bin/activate"
echo "   export \$(cat .env | xargs)"
echo ""
echo "2. Adicione um administrador:"
echo "   export FLASK_APP=servidor_pi.app"
echo "   flask add-admin admin senha123"
echo ""
echo "3. Inicie o servidor (desenvolvimento):"
echo "   python -m servidor_pi.app"
echo ""
echo "4. Ou inicie com Gunicorn (produção):"
echo "   gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 'servidor_pi.app:app'"
echo ""
echo "5. Descubra o IP do Pi:"
echo "   hostname -I"
echo ""
echo -e "${BLUE}Interface Web:${NC}"
echo "   Acesse http://<IP_DO_PI>:5000/web/"
echo ""
echo -e "${BLUE}Funcionalidades:${NC}"
echo "   • Dashboard de estatísticas"
echo "   • Gerenciamento de usuários"
echo "   • Detecção de intrusão"
echo "   • Backup automático (24h)"
echo ""
