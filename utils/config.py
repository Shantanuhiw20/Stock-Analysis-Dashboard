import os
from dotenv import load_dotenv

# load .env
load_dotenv()

# Embedding & model configs
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
LLM_MODEL        = os.getenv("LLM_MODEL", "llama3-8b-8192")
CHROMA_DIR       = os.getenv("CHROMA_DIR", "chroma_db")

# TF/HF env vars (example)
# TF_CPP_MIN_LOG_LEVEL=2
# TF_ENABLE_ONEDNN_OPTS=0
# HF_HUB_DISABLE_SYMLINKS_WARNING=1