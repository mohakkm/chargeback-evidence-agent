"""
Application Configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
ENV_FILE = Path(__file__).parents[1] / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()

# Groq API settings
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Qdrant settings
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

# Bounded Action Gate
AUTO_SUBMIT_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("AUTO_SUBMIT_CONFIDENCE_THRESHOLD", "0.80")
)
