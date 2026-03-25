"""
Module RAG Chain pour le système de recommandation d'événements culturels.

Ce module gère :
- Chargement de l'index FAISS
- Configuration du LLM Mistral
- Création de la chaîne RAG complète
- Interface de génération de réponses avec sources et confiance
"""

import os
import asyncio
from datetime import date
from pathlib import Path
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# ÉTAPE 1 : INITIALISATION DES VARIABLES D'ENVIRONNEMENT
# ============================================================================

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Vérifier que la clé API Mistral est présente
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError(
        "❌ La variable d'environnement MISTRAL_API_KEY est manquante. "
        "Vérifie ton fichier .env"
    )

# Définir le chemin racine du projet pour accéder au répertoire "rag"
PROJECT_ROOT = Path(__file__).parent.parent
VECTORSTORE_PATH = PROJECT_ROOT / "rag" / "vectorstore" / "faiss_index"


# ============================================================================
# ÉTAPE 2 : INITIALISATION DES EMBEDDINGS
# ============================================================================

# Utilise le même modèle d'embeddings que dans les scripts de préparation
# pour garantir la cohérence sémantique
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},  # Change en "cuda" si tu as une GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embeddings HuggingFace initialisés avec succès")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de l'initialisation des embeddings : {e}")


# ============================================================================
# ÉTAPE 3 : CHARGEMENT DE L'INDEX FAISS
# ============================================================================

try:
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ Index FAISS chargé avec succès depuis {VECTORSTORE_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"❌ L'index FAISS n'a pas été trouvé à l'emplacement : {VECTORSTORE_PATH}\n"
        "Vérifie qu'il a bien été créé avec 01_prepare_vector_store.py"
    )
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement de l'index FAISS : {e}")


# ============================================================================
# ÉTAPE 4 : CONFIGURATION DU RETRIEVER
# ============================================================================

# Crée un retriever MMR (Maximal Marginal Relevance) pour forcer la diversité
# des résultats : récupère 20 candidats, retient les 6 les plus pertinents ET
# les plus diversifiés. Evite de renvoyer 4 chunks du même lieu (ex: Meudon).
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 50, "lambda_mult": 0.7}
)
print("✅ Retriever MMR configuré (k=6, fetch_k=20) pour diversité géographique")


# ============================================================================
# ÉTAPE 5 : INITIALISATION DU MODÈLE DE LANGAGE
# ============================================================================

try:
    # Conserve le même LLM Mistral pour ne rien changer au comportement métier.
    # Ce choix est parfait pour le POC : on réutilise la configuration déjà validée.
    mistral_llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.3,
        api_key=MISTRAL_API_KEY,
        max_tokens=2048
    )
    print("✅ Modèle ChatMistralAI initialisé (mistral-large-latest)")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de l'initialisation du LLM : {e}")


# ============================================================================
# ÉTAPE 6 : DÉFINITION DU PROMPT TEMPLATE DE RÉPONSE
# ============================================================================

# Garde un prompt simple, lisible et entièrement portable.
# C'est idéal pour le POC : zéro dépendance nouvelle, zéro stockage externe,
# et une consigne claire pour rester fidèle au contexte RAG.
_MONTHS_FR = ["janvier","février","mars","avril","mai","juin",
              "juillet","août","septembre","octobre","novembre","décembre"]
_today = date.today()
_today_str = f"{_today.day} {_MONTHS_FR[_today.month - 1]} {_today.year}"

system_prompt = f"""Tu es un assistant expert en événements culturels pour Grand Paris Seine Ouest (GPSO).
Aujourd'hui, nous sommes le {_today_str}.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT en te basant sur le contexte fourni. Ne fabrique aucun événement.
2. Si la réponse n'est pas dans le contexte, dis clairement : "Je n'ai pas trouvé d'événement correspondant dans ma base."
3. Pour chaque événement cité, indique toujours : le nom, la date, l'heure, le lieu précis, le tarif et les conditions d'inscription si disponibles.
4. Utilise la date d'aujourd'hui pour interpréter "ce week-end", "ce mois-ci", "prochainement".
5. Réponds en français, de façon structurée et concise (liste à puces si plusieurs événements).
6. Ne reformule pas la question, va directement à la réponse."""

# Utilise le placeholder {question} attendu par ConversationalRetrievalChain.
# Cela permet d'intégrer la mémoire sans casser le flux de la chaîne standard.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Contexte : {context}

Question : {question}""")
])
print("✅ Prompt conversationnel en français configuré")


# ============================================================================
# ÉTAPE 7 : INITIALISATION DE LA MÉMOIRE COURT-TERME
# ============================================================================

try:
    # Stocke l'historique complet de la discussion en mémoire vive.
    # Ce choix est parfait pour le POC : c'est portable, léger et très rapide.
    memory = ConversationBufferMemory(
        # Utilise exactement la clé attendue par la chaîne conversationnelle.
        memory_key="chat_history",
        # Fixe explicitement la clé d'entrée pour éviter toute ambiguïté.
        input_key="question",
        # Fixe explicitement la clé de sortie pour mémoriser uniquement la réponse.
        output_key="answer",
        # Retourne des messages structurés, ce qui convient aux chat models modernes.
        return_messages=True,
    )
    print("✅ Mémoire ConversationBufferMemory initialisée en RAM")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de l'initialisation de la mémoire : {e}")


# ============================================================================
# ÉTAPE 8 : CRÉATION DE LA CHAÎNE RAG CONVERSATIONNELLE
# ============================================================================

try:
    # Remplace la chaîne stateless par une chaîne conversationnelle native.
    # C'est parfait pour le POC : une seule abstraction ajoute l'historique,
    # reformule les questions de suivi et reste très simple à maintenir.
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        # Réutilise exactement le LLM Mistral déjà validé dans le projet.
        llm=mistral_llm,
        # Réutilise exactement le retriever FAISS en limitant le contexte à 4 docs.
        retriever=retriever,
        # Branche la mémoire courte pour que la chaîne se souvienne du fil courant.
        memory=memory,
        # Réinjecte le prompt métier existant pour conserver la qualité des réponses.
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        # Retourne aussi les documents utilisés pour reconstruire les sources API.
        return_source_documents=True,
        # Active les logs internes pour faciliter la démo et le debug.
        verbose=True,
    )

    print("✅ Chaîne RAG conversationnelle créée avec succès")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de la création de la chaîne RAG : {e}")


# ============================================================================
# ÉTAPE 9 : FONCTION DE GÉNÉRATION DE RÉPONSE
# ============================================================================

async def generate_response(
    question: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Génère une réponse complète avec sources et confiance.
    
    Args:
        question (str): La question posée par l'utilisateur
        filters (dict, optional): Filtres optionnels pour la recherche.
                                 À implémenter ultérieurement.
    
    Returns:
        dict: Structure de réponse contenant :
            - "answer" (str): Réponse textuelle du LLM
            - "sources" (list[dict]): Événements sources utilisés
            - "confidence" (float): Score de confiance (0.0 à 1.0)
    
    Exemple de réponse :
        {
            "answer": "Voici les événements à venir...",
            "sources": [
                {"title": "Concert Jazz", "location": "Boulogne", "date": "2026-04-15"},
                ...
            ],
            "confidence": 0.85
        }
    """
    
    if not question or not question.strip():
        raise ValueError("❌ La question ne peut pas être vide")
    
    try:
        # Appelle la chaîne conversationnelle dans un thread pour ne pas bloquer l'API.
        # On passe uniquement la question : l'historique est injecté automatiquement.
        chain_result = await asyncio.to_thread(
            retrieval_chain.invoke,
            {"question": question}
        )
        
        # Récupère le texte final dans la clé standard "answer".
        # Cela garde un mapping simple et stable pour le reste de l'application.
        answer_text = str(chain_result.get("answer", ""))

        # Récupère exactement les documents qui ont servi à générer la réponse.
        # Ce choix est meilleur qu'un second retrieve sur la question brute,
        # surtout pour les relances du type "Et il y en a d'autres ?".
        docs = chain_result.get("source_documents", [])
        
        # Extrait les métadonnées des sources
        sources: List[Dict[str, str]] = []
        for doc in docs:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            
            # Crée une entrée source structurée
            # Note : utilise 'timings' à la place de 'date' (c'est le champ réel dans les métadonnées)
            source_item = {
                "title": metadata.get("title", "Sans titre"),
                "location": metadata.get("location", "Lieu non spécifié"),
                "date": metadata.get("timings", "Date non spécifiée")
            }
            sources.append(source_item)
        
        # Calcule une confiance simple
        # 0.85 si des sources ont été trouvées, 0.3 sinon
        confidence = 0.85 if sources else 0.3
        
        # Retourne la structure de réponse
        response = {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence
        }
        
        return response
    
    except ValueError as ve:
        # Erreur de validation
        raise ValueError(f"❌ Erreur de validation : {ve}")
    
    except Exception as e:
        # Erreur d'exécution
        raise RuntimeError(
            f"❌ Erreur lors de la génération de la réponse : {e}"
        )


# ============================================================================
# FONCTION SYNCHRONE (wrapper)
# ============================================================================

def generate_response_sync(
    question: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Wrapper synchrone pour appeler generate_response.
    Utile si tu dois l'utiliser dans un contexte non-async.
    
    Args:
        question (str): La question
        filters (dict, optional): Filtres optionnels
    
    Returns:
        dict: Structure de réponse
    """
    try:
        # Gère les cas où une boucle d'événements existe déjà
        try:
            loop = asyncio.get_running_loop()
            # Si on est déjà dans une loop (ex. Jupyter, Uvicorn), 
            # utiliser un thread pour ne pas bloquer
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, generate_response(question, filters))
                return future.result()
        except RuntimeError:
            # Pas de loop en cours, on peut créer une nouvelle
            return asyncio.run(generate_response(question, filters))
    except Exception as e:
        raise RuntimeError(f"❌ Erreur dans le wrapper synchrone : {e}")


# ============================================================================
# MESSAGE D'INITIALISATION
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     🎭 SYSTÈME RAG READY FOR ACTION 🎭                     ║
╠════════════════════════════════════════════════════════════════════════════╣
║ ✅ Index FAISS chargé et configuré                                         ║
║ ✅ Retriever initialisé (k=4)                                              ║
║ ✅ LLM Mistral Large configuré (T=0.3)                                      ║
║ ✅ Mémoire courte ConversationBufferMemory active                          ║
║ ✅ Chaîne RAG conversationnelle opérationnelle                             ║
║ ✅ Fonction generate_response prête à l'emploi                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Utilisation :
    from app.rag_chain import generate_response_sync
    response = generate_response_sync("Quel événement y a-t-il à Boulogne?")
    print(response["answer"])
    print(response["sources"])
    print(response["confidence"])

    # Ou en async :
    import asyncio
    from app.rag_chain import generate_response
    result = asyncio.run(generate_response("Ma question"))
""")
