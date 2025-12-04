"""
Módulo de segurança e detecção de tentativas de invasão.
Monitora tentativas de acesso falhas e implementa bloqueios temporários.
"""
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta

# Configurações de segurança
MAX_FAILED_ATTEMPTS = 5  # Máximo de tentativas antes do bloqueio
BLOCK_DURATION_MINUTES = 15  # Duração do bloqueio em minutos
ATTEMPT_WINDOW_MINUTES = 10  # Janela de tempo para contar tentativas

# Armazenamento em memória (thread-safe)
_lock = threading.Lock()
_failed_attempts = defaultdict(list)  # {ip: [timestamps]}
_blocked_ips = {}  # {ip: unblock_time}
_access_log = []  # Lista de todos os acessos para estatísticas
_alerts = []  # Alertas de segurança


class SecurityEvent:
    """Representa um evento de segurança."""
    def __init__(self, event_type, ip_address, details="", user_name=None):
        self.timestamp = datetime.now()
        self.event_type = event_type  # 'access_granted', 'access_denied', 'blocked', 'alert'
        self.ip_address = ip_address
        self.details = details
        self.user_name = user_name
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "ip_address": self.ip_address,
            "details": self.details,
            "user_name": self.user_name
        }


def is_ip_blocked(ip_address):
    """Verifica se um IP está bloqueado."""
    with _lock:
        if ip_address in _blocked_ips:
            if datetime.now() < _blocked_ips[ip_address]:
                return True
            else:
                # Bloqueio expirou
                del _blocked_ips[ip_address]
                return False
        return False


def get_block_remaining_time(ip_address):
    """Retorna tempo restante de bloqueio em segundos."""
    with _lock:
        if ip_address in _blocked_ips:
            remaining = (_blocked_ips[ip_address] - datetime.now()).total_seconds()
            return max(0, int(remaining))
        return 0


def record_failed_attempt(ip_address, details=""):
    """Registra uma tentativa de acesso falha."""
    with _lock:
        now = datetime.now()
        cutoff = now - timedelta(minutes=ATTEMPT_WINDOW_MINUTES)
        
        # Remove tentativas antigas
        _failed_attempts[ip_address] = [
            t for t in _failed_attempts[ip_address] if t > cutoff
        ]
        
        # Adiciona nova tentativa
        _failed_attempts[ip_address].append(now)
        
        # Registra evento
        event = SecurityEvent('access_denied', ip_address, details)
        _access_log.append(event)
        
        # Verifica se deve bloquear
        if len(_failed_attempts[ip_address]) >= MAX_FAILED_ATTEMPTS:
            block_ip(ip_address)
            return True
        
        return False


def record_successful_access(ip_address, user_name):
    """Registra um acesso bem-sucedido."""
    with _lock:
        # Limpa tentativas falhas anteriores
        if ip_address in _failed_attempts:
            del _failed_attempts[ip_address]
        
        # Registra evento
        event = SecurityEvent('access_granted', ip_address, user_name=user_name)
        _access_log.append(event)


def block_ip(ip_address):
    """Bloqueia um IP por tentativas excessivas."""
    with _lock:
        unblock_time = datetime.now() + timedelta(minutes=BLOCK_DURATION_MINUTES)
        _blocked_ips[ip_address] = unblock_time
        
        # Limpa tentativas
        if ip_address in _failed_attempts:
            del _failed_attempts[ip_address]
        
        # Registra alerta
        alert = SecurityEvent(
            'alert',
            ip_address,
            f"IP bloqueado por {BLOCK_DURATION_MINUTES} minutos devido a {MAX_FAILED_ATTEMPTS} tentativas falhas"
        )
        _alerts.append(alert)
        _access_log.append(SecurityEvent('blocked', ip_address))
        
        print(f"[SECURITY] IP {ip_address} bloqueado até {unblock_time}")


def unblock_ip(ip_address):
    """Remove bloqueio de um IP manualmente."""
    with _lock:
        if ip_address in _blocked_ips:
            del _blocked_ips[ip_address]
            return True
        return False


def get_blocked_ips():
    """Retorna lista de IPs bloqueados."""
    with _lock:
        now = datetime.now()
        blocked = []
        for ip, unblock_time in list(_blocked_ips.items()):
            if now < unblock_time:
                blocked.append({
                    "ip": ip,
                    "unblock_at": unblock_time.isoformat(),
                    "remaining_seconds": int((unblock_time - now).total_seconds())
                })
            else:
                del _blocked_ips[ip]
        return blocked


def get_recent_alerts(limit=50):
    """Retorna alertas recentes."""
    with _lock:
        return [a.to_dict() for a in _alerts[-limit:]]


def get_access_statistics(hours=24):
    """Retorna estatísticas de acesso das últimas N horas."""
    with _lock:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [e for e in _access_log if e.timestamp > cutoff]
        
        stats = {
            "total_attempts": len(recent),
            "successful": sum(1 for e in recent if e.event_type == 'access_granted'),
            "denied": sum(1 for e in recent if e.event_type == 'access_denied'),
            "blocked": sum(1 for e in recent if e.event_type == 'blocked'),
            "by_hour": defaultdict(lambda: {"granted": 0, "denied": 0}),
            "by_user": defaultdict(int),
            "by_ip": defaultdict(lambda: {"granted": 0, "denied": 0})
        }
        
        for event in recent:
            hour = event.timestamp.strftime("%Y-%m-%d %H:00")
            
            if event.event_type == 'access_granted':
                stats["by_hour"][hour]["granted"] += 1
                if event.user_name:
                    stats["by_user"][event.user_name] += 1
                stats["by_ip"][event.ip_address]["granted"] += 1
            elif event.event_type == 'access_denied':
                stats["by_hour"][hour]["denied"] += 1
                stats["by_ip"][event.ip_address]["denied"] += 1
        
        # Converte defaultdicts para dicts normais
        stats["by_hour"] = dict(stats["by_hour"])
        stats["by_user"] = dict(stats["by_user"])
        stats["by_ip"] = dict(stats["by_ip"])
        
        return stats


def get_access_log(limit=100):
    """Retorna log de acessos recentes."""
    with _lock:
        return [e.to_dict() for e in _access_log[-limit:]]


def cleanup_old_logs(days=7):
    """Remove logs antigos para economizar memória."""
    global _access_log, _alerts
    with _lock:
        cutoff = datetime.now() - timedelta(days=days)
        _access_log = [e for e in _access_log if e.timestamp > cutoff]
        _alerts = [a for a in _alerts if a.timestamp > cutoff]
        return True
