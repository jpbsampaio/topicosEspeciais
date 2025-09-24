#!/usr/bin/env python3
"""
Configurações centrais do projeto.

Lê variáveis de ambiente (via .env, se presente) e expõe constantes usadas por servidor/cliente/handlers.
"""

from __future__ import annotations

import os
from typing import Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv é opcional; se não estiver instalado, seguimos com os defaults/ambiente
    pass


# Redes
SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8888"))
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "5"))

# Câmera
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "480"))

# Paths
MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
DATA_DIR: str = os.getenv("DATA_DIR", "data")
LOG_DIR: str = os.getenv("LOG_DIR", "logs")

# Reconhecimento (LBPH)
LBPH_THRESHOLD: float = float(os.getenv("LBPH_THRESHOLD", "70.0"))


def camera_resolution() -> Tuple[int, int]:
    return (CAMERA_WIDTH, CAMERA_HEIGHT)
