"""Vercel ASGI entry point; local development continues to use ml-service/app.py."""
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1] / "ml-service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi import FastAPI  # noqa: E402
from app import app as ml_app  # noqa: E402

# Vercel forwards the original request path to this ASGI application. Mounting
# preserves local ml-service routes (/infer, /health) while serving production
# requests under /api/ml/infer and /api/ml/health.
app = FastAPI(title="EyeZen ML Vercel Adapter")
app.mount("/api/ml", ml_app)
