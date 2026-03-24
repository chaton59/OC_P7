from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn
from pathlib import Path
import time
import subprocess
import sys
from app.models import QuestionRequest, AnswerResponse
from app.rag_chain import generate_response

# Chargement automatique du .env au démarrage
load_dotenv()

# Configuration de l'application FastAPI
app = FastAPI(
    title="Assistant RAG Événements Culturels GPSO",
    description="API POC pour recommandation d'événements via Mistral + Faiss",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Interface de chat
@app.get("/", tags=["Chat"], summary="Interface de chat")
def chat_ui():
    """Sert l'interface de chat HTML."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# Route de statut (health check)
@app.get("/health", tags=["Status"], summary="Vérification du statut API")
def health_check():
    return {"status": "ok", "message": "API RAG prête"}


# Endpoint RAG pour poser une question
@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["RAG"],
    summary="Poser une question à l'assistant RAG"
)
async def ask_question(request: QuestionRequest):
    """
    Endpoint pour poser une question à l'assistant RAG.
    
    L'endpoint effectue une recherche sémantique sur la base de données vectorielle (FAISS)
    et utilise le modèle Mistral pour générer une réponse contextuelle.
    
    Args:
        request: Requête contenant la question et les filtres optionnels
        
    Returns:
        AnswerResponse: Réponse avec la génération, les sources et le score de confiance
    """
    try:
        result = await generate_response(request.question, request.filters)
        return result
    except Exception as e:
        print(f"❌ Erreur RAG : {e}")
        return {
            "answer": f"Désolé, une erreur est survenue : {str(e)}",
            "sources": [],
            "confidence": 0.0
        }



# Fonction pour exécuter la reconstruction en background
def _run_rebuild():
    """Exécute la reconstruction de l'index en background."""
    try:
        print("🔄 Reconstruction de l'index demandée...")
        start_time = time.time()
        
        # OK 6.3 - Exécution du script de préparation du vectorstore
        script_path = Path(__file__).parent / "scripts" / "01_prepare_vector_store.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = result.stderr or "Erreur inconnue lors de la reconstruction"
            print(f"❌ Erreur reconstruction : {error_msg}")
            return
        
        # Parsing du stdout pour extraire le nombre de documents indexés
        docs_indexed = 0
        for line in result.stdout.split("\n"):
            if "Nombre total de documents indexés" in line:
                try:
                    docs_indexed = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
        
        print(f"✅ Index reconstruit en {duration:.2f}s avec {docs_indexed} documents")
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout: la reconstruction a dépassé 600 secondes")
    except Exception as e:
        print(f"❌ Erreur /rebuild en background : {e}")


# Endpoint pour reconstruction de l'index
@app.post("/rebuild", tags=["Admin"], status_code=202)
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Lance la reconstruction de l'index vectoriel FAISS à partir de clean_events.csv.
    Exécute le script 01_prepare_vector_store.py en background task.

    Returns:
        dict: Status avec informations de reconstruction
    """
    # Lancer la reconstruction en background (OK 6.3 - étape reconstruction index)
    background_tasks.add_task(_run_rebuild)

    return {
        "status": "accepted",
        "message": "Index vectoriel en cours de reconstruction (processus background)...",
        "reload_required": True
    }


# Point d'entrée pour démarrage avec uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
