"""Vercel ASGI entry point; local development continues to use ml-service/app.py."""
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1] / "ml-service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from app import app  # noqa: E402,F401
