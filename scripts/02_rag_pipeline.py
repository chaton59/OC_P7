"""
Configuration Mistral

Étape 4.1 : Configuration du modèle de langage Mistral pour le pipeline RAG.

Ce script initialise et configure le modèle Mistral AI avec les paramètres
optimisés pour le projet de recommandation d'événements culturels.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Charger les variables d'environnement depuis le fichier .env
# Cela permet de sécuriser les clés API sans les hardcoder dans le code
load_dotenv()

# Définir la racine du projet pour accéder au répertoire "rag"
PROJECT_ROOT = Path(__file__).parent.parent

# Initialisation du modèle de langage Mistral
# Nous utilisons ChatMistralAI pour les interactions conversationnelles
llm = ChatMistralAI(
    model="mistral-small-latest",  # Modèle small : bon compromis entre coût et performance
    temperature=0.2,               # Basse température (0.2) pour des réponses déterministes et factuelles
    api_key=os.getenv("MISTRAL_API_KEY")  # Clé API récupérée depuis les variables d'environnement
)


# ============================================================================
# SOUS-ÉTAPE 4.2 : Prompt système optimisé et structuration des réponses
# ============================================================================

# Définition du prompt système optimisé
# Ce prompt configure l'assistant pour répondre de manière structurée et contextuelle.
# Les 8 communes du Grand Paris Seine Ouest : Boulogne-Billancourt, Chaville,
# Issy-les-Moulineaux, Marnes-la-Coquette, Meudon, Sèvres, Vanves et Ville-d'Avray.
system_prompt = """Tu es un assistant expert spécialisé dans les événements culturels du Grand Paris Seine Ouest.

📋 RÈGLES DE RÉPONSE (STRICTES) :
1. Réponds UNIQUEMENT avec les informations fournies dans le contexte.
2. Ne pas inventer d'événements ou de détails non mentionnés.
3. Structure chaque réponse de manière claire et agréable selon ce modèle :
   • 🎭 Titre de l'événement
   • 📅 Date et horaire
   • 📍 Lieu
   • 📝 Description courte (2-3 lignes maximum)
   • 💰 Conditions (gratuit/payant, catégorie d'âge, inscription requise)
   • 🔗 Lien si disponible

4. Si aucun événement ne correspond à la recherche, réponds poliment :
   "Aucun événement ne correspond à ta recherche dans notre base de données.
   Peux-tu me poser une autre question ou affiner ta recherche ?"

5. Réponds TOUJOURS en français.
6. N'hésite pas à suggérer des événements, à aider l'utilisateur à découvrir des activités auxquelles il n'aurait pas pensé.

🎯 OBJECTIF : Aider les utilisateurs à découvrir des événements culturels pertinents
de manière claire, structurée et basée uniquement sur les données disponibles."""

# Création du modèle de prompt avec ChatPromptTemplate
# Ce modèle combine :
# - Un prompt système qui définit le comportement et les règles de l'assistant
# - Un placeholder {input} pour la question de l'utilisateur
#
# POURQUOI ce prompt est strict et structuré ?
# 1. Éviter les hallucinations : en restreignant à UNIQUEMENT le contexte fourni
# 2. Garantir la pertinence : les utilisateurs reçoivent exactement ce qu'ils cherchent
# 3. Améliorer la UX : une structure claire et prévisible inspire confiance
# 4. Faciliter le RAG : le modèle sait exactement comment présenter les résultats retrouvés
# 5. Fiabilité : pas de données erronées ou obsolètes
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", """Contexte (événements disponibles) :\n{context}\n\nQuestion utilisateur : {input}\n\nRéponds avec les événements qui correspondent à la question, en respectant la structure définie dans tes instructions système.""")
])


# ============================================================================
# SOUS-ÉTAPE 4.3 - CHARGEMENT VECTOR STORE + RETRIEVER
# ============================================================================

def create_retriever():
    """Charge le FAISS vectorstore de l'étape 3 et retourne un retriever prêt à l'emploi.
    
    Choix pédagogiques :
    - Même modèle d'embeddings que l'étape 3 → cohérence sémantique parfaite
    - k=4 → 4 résultats les plus pertinents (équilibre qualité/vitesse)
    - allow_dangerous_deserialization=True → obligatoire depuis LangChain 0.2+
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Chemin exact du vector store créé à l'étape 3
    vectorstore_path = PROJECT_ROOT / "rag" / "vectorstore" / "faiss_index"
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever


# Création du retriever (une seule fois au démarrage)
retriever = create_retriever()


# ============================================================================
# SOUS-ÉTAPE 4.4 - CHAÎNE RAG COMPLÈTE
# ============================================================================

# Format du prompt avec le contexte du retriever
def format_docs(docs):
    """Formate les documents récupérés en contexte pour le prompt."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Chaîne RAG construite avec LCEL (méthode compatible LangChain 1.2)
# Composée de : retriever → format docs → prompt + LLM
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)

def get_recommendation(question: str) -> str:
    """Fonction principale du RAG : question → réponse augmentée.
    
    Choix pédagogiques :
    - Utilise LCEL (LangChain Expression Language) recommandé
    - Simple, composable et prêt pour l'API (étape 5)
    """
    if not question or not question.strip():
        return "❌ Veuillez poser une vraie question sur les événements culturels !"
    
    response = rag_chain.invoke(question)
    return response.content


if __name__ == "__main__":
    # ========================================================================
    # TESTS FINAUX DU RAG (SOUS-ÉTAPE 4.4)
    # ========================================================================
    print("✅ Pipeline RAG configuré avec succès !")
    print("\n🎯 TESTS FINAUX DU PIPELINE RAG")
    
    # Tests réalistes
    tests = [
        "Quelles soirées chouettes sont prévues ce mois-ci ?",
        "Y a-t-il des ateliers vélo à Boulogne-Billancourt ou Vanves ?",
        "Recommande-moi un concert de piano près de Meudon"
    ]
    
    for q in tests:
        print(f"\n❓ Question : {q}")
        print(get_recommendation(q))
        print("-" * 80)
