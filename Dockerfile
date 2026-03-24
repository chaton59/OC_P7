# ============================================================================
# Multi-stage Dockerfile pour API FastAPI RAG - Événements Culturels
# Optimisé pour légèreté et performance
# ============================================================================

# --- STAGE 1: Builder ---
FROM python:3.12-slim AS builder

# Installer UV depuis les sources officielles
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ajouter UV au PATH
ENV PATH="/root/.local/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /build

# Copier les fichiers de dépendances
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# Exporter les dépendances figées dans un requirements.txt
# Cela garantit la reproductibilité avec uv.lock
RUN uv export --frozen --format requirements-txt > requirements.txt

# --- STAGE 2: Runtime ---
FROM python:3.12-slim AS runtime

# Définir le répertoire de travail
WORKDIR /app

# Copier le requirements.txt depuis le builder
COPY --from=builder /build/requirements.txt requirements.txt

# Installer les dépendances directement dans le Python système
# (approche simple et fiable)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY main.py .
COPY app/ ./app/
COPY rag/ ./rag/
COPY static/ ./static/
# OK 6.3 - scripts de reconstruction de l'index
COPY scripts/ ./scripts/
# donnees source pour le rebuild
COPY data/ ./data/

# Utilisateur non-root pour la sécurité
RUN useradd --create-home --user-group appuser && chown -R appuser:appuser /app
USER appuser

# Variables d'environnement
ENV PYTHONUNBUFFERED=1

# Exposer le port 8000
EXPOSE 8000

# Healthcheck pour Kubernetes/Docker Compose
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/docs')" || exit 1

# Démarrer l'application avec uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
