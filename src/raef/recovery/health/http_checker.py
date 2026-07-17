import requests
from .base import HealthChecker

class HttpHealthChecker(HealthChecker):
    def __init__(self, url: str, timeout: int = 2):
        self.url = url
        self.timeout = timeout

    def is_alive(self) -> bool:
        try:
            resp = requests.get(self.url, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False