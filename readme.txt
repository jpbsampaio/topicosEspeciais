5.2. Teste e Implantação End-to-End
Siga estas etapas para executar o sistema completo em uma rede local (LAN).

Passo 1: Preparar o Servidor (no Raspberry Pi)

Navegue até a pasta servidor_pi/.

Crie um ambiente virtual: python3 -m venv venv e source venv/bin/activate.

Instale as dependências: pip install -r requirements.txt.

Inicialize o banco de dados: flask init-db. Isso criará o arquivo database.db.

Adicione o primeiro administrador (substitua com suas credenciais): flask add-admin seu_admin sua_senha_segura

Passo 2: Iniciar o Servidor de Produção (no Raspberry Pi) O servidor de desenvolvimento do Flask (flask run) não deve ser usado, pois é de thread único. O requisito de ThreadPool (Imagem 2) é satisfeito pelo Gunicorn (RNF-06).   

Execute o Gunicorn: gunicorn --workers 4 --bind 0.0.0.0:5000 "servidor_pi.app:app"

--workers 4: Cria um "pool" de 4 processos de workers para lidar com solicitações concorrentes.   

--bind 0.0.0.0:5000: Instrui o Gunicorn a escutar em todas as interfaces de rede na porta 5000, permitindo que outros dispositivos (os clientes) na sua rede Wi-Fi se conectem a ele.

"servidor_pi.app:app": Informa ao Gunicorn para carregar a variável app do módulo servidor_pi.app.

Passo 3: Configurar e Executar o Cliente Admin (no seu Notebook)

Obtenha o endereço IP do seu Raspberry Pi (ex: 192.168.1.10).

Navegue até a pasta cliente_admin/.

Crie e ative um ambiente virtual e instale as dependências: pip install -r requirements.txt.

Edite o arquivo cliente_admin/admin.py e mude a variável BASE_URL para o IP do seu Pi: BASE_URL = "http://192.168.1.10:5000"

Execute o cliente: python admin.py.

Faça login com as credenciais criadas no Passo 1.

Use a opção "1. Adicionar Usuario" para cadastrar seu rosto e o de outras pessoas.

Passo 4: Configurar e Executar o Cliente Kiosk (no seu Notebook ou outro PC)

Navegue até a pasta cliente_acesso/.

Instale as dependências: pip install -r requirements.txt.

Edite cliente_acesso/kiosk.py e configure o BASE_URL para o IP do Pi: BASE_URL = "http://192.168.1.10:5000"

Execute o cliente: python kiosk.py.

Uma janela da webcam aparecerá. Pressione a barra de ESPAÇO.

Se tudo estiver configurado corretamente, o Kiosk mostrará "Verificando...", enviará a imagem para o Pi, o Gunicorn em execução no Pi encaminhará a solicitação para um worker, o deepface e o numpy encontrarão a correspondência, e o servidor retornará {"acesso_liberado": True}. O Kiosk então exibirá a imagem unlocked.png por 5 segundos.