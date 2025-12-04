"""
Módulo de backup automático do banco de dados e modelo LBPH.
Suporta backups agendados e manuais.
"""
import os
import shutil
import threading
import time
from datetime import datetime
import sqlite3

# Configurações de backup
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'backups')
MAX_BACKUPS = 10  # Número máximo de backups mantidos
BACKUP_INTERVAL_HOURS = 24  # Intervalo entre backups automáticos

# Arquivos a fazer backup
DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'database.db')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

_backup_thread = None
_stop_backup = threading.Event()


def ensure_backup_dir():
    """Garante que o diretório de backup existe."""
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_backup(manual=False):
    """
    Cria um backup completo do sistema.
    
    Args:
        manual: Se True, marca o backup como manual
    
    Returns:
        dict com informações do backup criado
    """
    ensure_backup_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_type = "manual" if manual else "auto"
    backup_name = f"backup_{backup_type}_{timestamp}"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    
    try:
        os.makedirs(backup_path)
        
        files_backed_up = []
        
        # Backup do banco de dados
        if os.path.exists(DATABASE_FILE):
            db_backup = os.path.join(backup_path, 'database.db')
            # Usa conexão para garantir consistência
            conn = sqlite3.connect(DATABASE_FILE)
            backup_conn = sqlite3.connect(db_backup)
            conn.backup(backup_conn)
            backup_conn.close()
            conn.close()
            files_backed_up.append('database.db')
        
        # Backup do modelo LBPH
        model_file = os.path.join(MODEL_DIR, 'lbph_model.yml')
        labels_file = os.path.join(MODEL_DIR, 'labels.pkl')
        
        if os.path.exists(model_file):
            shutil.copy2(model_file, os.path.join(backup_path, 'lbph_model.yml'))
            files_backed_up.append('lbph_model.yml')
        
        if os.path.exists(labels_file):
            shutil.copy2(labels_file, os.path.join(backup_path, 'labels.pkl'))
            files_backed_up.append('labels.pkl')
        
        # Cria arquivo de metadados
        metadata = {
            "timestamp": timestamp,
            "type": backup_type,
            "files": files_backed_up
        }
        
        with open(os.path.join(backup_path, 'metadata.txt'), 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        # Remove backups antigos
        cleanup_old_backups()
        
        print(f"[BACKUP] Criado: {backup_name} ({len(files_backed_up)} arquivos)")
        
        return {
            "success": True,
            "backup_name": backup_name,
            "files": files_backed_up,
            "path": backup_path
        }
        
    except Exception as e:
        print(f"[BACKUP] Erro ao criar backup: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def list_backups():
    """Lista todos os backups disponíveis."""
    ensure_backup_dir()
    
    backups = []
    
    for name in sorted(os.listdir(BACKUP_DIR), reverse=True):
        backup_path = os.path.join(BACKUP_DIR, name)
        if os.path.isdir(backup_path) and name.startswith('backup_'):
            # Lê metadados se existir
            metadata_file = os.path.join(backup_path, 'metadata.txt')
            metadata = {}
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()
            
            # Calcula tamanho
            size = sum(
                os.path.getsize(os.path.join(backup_path, f))
                for f in os.listdir(backup_path)
                if os.path.isfile(os.path.join(backup_path, f))
            )
            
            backups.append({
                "name": name,
                "timestamp": metadata.get("timestamp", "unknown"),
                "type": metadata.get("type", "unknown"),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2)
            })
    
    return backups


def restore_backup(backup_name):
    """
    Restaura um backup específico.
    
    Args:
        backup_name: Nome do backup a restaurar
    
    Returns:
        dict com resultado da restauração
    """
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    
    if not os.path.exists(backup_path):
        return {"success": False, "error": "Backup não encontrado"}
    
    try:
        restored = []
        
        # Restaura banco de dados
        db_backup = os.path.join(backup_path, 'database.db')
        if os.path.exists(db_backup):
            shutil.copy2(db_backup, DATABASE_FILE)
            restored.append('database.db')
        
        # Restaura modelo LBPH
        model_backup = os.path.join(backup_path, 'lbph_model.yml')
        labels_backup = os.path.join(backup_path, 'labels.pkl')
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if os.path.exists(model_backup):
            shutil.copy2(model_backup, os.path.join(MODEL_DIR, 'lbph_model.yml'))
            restored.append('lbph_model.yml')
        
        if os.path.exists(labels_backup):
            shutil.copy2(labels_backup, os.path.join(MODEL_DIR, 'labels.pkl'))
            restored.append('labels.pkl')
        
        print(f"[BACKUP] Restaurado: {backup_name} ({len(restored)} arquivos)")
        
        return {
            "success": True,
            "backup_name": backup_name,
            "restored_files": restored
        }
        
    except Exception as e:
        print(f"[BACKUP] Erro ao restaurar: {e}")
        return {"success": False, "error": str(e)}


def delete_backup(backup_name):
    """Remove um backup específico."""
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    
    if not os.path.exists(backup_path):
        return {"success": False, "error": "Backup não encontrado"}
    
    try:
        shutil.rmtree(backup_path)
        print(f"[BACKUP] Removido: {backup_name}")
        return {"success": True, "deleted": backup_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cleanup_old_backups():
    """Remove backups antigos mantendo apenas MAX_BACKUPS."""
    ensure_backup_dir()
    
    backups = sorted([
        d for d in os.listdir(BACKUP_DIR)
        if os.path.isdir(os.path.join(BACKUP_DIR, d)) and d.startswith('backup_')
    ])
    
    # Remove backups excedentes (mantém os mais recentes)
    while len(backups) > MAX_BACKUPS:
        oldest = backups.pop(0)
        backup_path = os.path.join(BACKUP_DIR, oldest)
        try:
            shutil.rmtree(backup_path)
            print(f"[BACKUP] Removido backup antigo: {oldest}")
        except Exception as e:
            print(f"[BACKUP] Erro ao remover {oldest}: {e}")


def _backup_worker():
    """Worker thread para backups automáticos."""
    while not _stop_backup.is_set():
        # Aguarda intervalo ou sinal de parada
        _stop_backup.wait(timeout=BACKUP_INTERVAL_HOURS * 3600)
        
        if not _stop_backup.is_set():
            create_backup(manual=False)


def start_auto_backup():
    """Inicia o sistema de backup automático."""
    global _backup_thread
    
    if _backup_thread is not None and _backup_thread.is_alive():
        print("[BACKUP] Sistema de backup já está rodando")
        return False
    
    _stop_backup.clear()
    _backup_thread = threading.Thread(target=_backup_worker, daemon=True)
    _backup_thread.start()
    
    print(f"[BACKUP] Sistema automático iniciado (intervalo: {BACKUP_INTERVAL_HOURS}h)")
    return True


def stop_auto_backup():
    """Para o sistema de backup automático."""
    global _backup_thread
    
    _stop_backup.set()
    
    if _backup_thread is not None:
        _backup_thread.join(timeout=5)
        _backup_thread = None
    
    print("[BACKUP] Sistema automático parado")
    return True


def get_backup_status():
    """Retorna status do sistema de backup."""
    return {
        "auto_backup_running": _backup_thread is not None and _backup_thread.is_alive(),
        "interval_hours": BACKUP_INTERVAL_HOURS,
        "max_backups": MAX_BACKUPS,
        "backup_dir": BACKUP_DIR,
        "total_backups": len(list_backups())
    }
