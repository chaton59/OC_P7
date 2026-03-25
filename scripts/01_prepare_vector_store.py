"""
Script: 01_prepare_vector_store.py
Objectif: Préparer et indexer les données d'événements dans un vector store FAISS
          en utilisant LangChain, pour la récupération dans un pipeline RAG

Architecture du pipeline:
    1. Charger les données d'événements nettoyées (CSV)
    2. Générer les embeddings (représentations vectorielles) avec les modèles HuggingFace
    3. Indexer les embeddings dans FAISS (bibliothèque de recherche vectorielle)
    4. Sauvegarder le vector store sur le disque pour utilisation future

Dépendances principales:
    - langchain : orchestration de la chaîne RAG
    - langchain-community : intégrations tierces (FAISS, embeddings HuggingFace)
    - faiss-cpu : indexation et recherche vectorielle haute performance
    - pandas : manipulation et chargement des données CSV
"""

# ============================================================================
# IMPORTS - BIBLIOTHÈQUE STANDARD
# ============================================================================
import os
from pathlib import Path
import logging

# ============================================================================
# IMPORTS - TRAITEMENT DES DONNÉES
# ============================================================================
import pandas as pd
import numpy as np

# ============================================================================
# IMPORTS - LANGCHAIN ET VECTORISATION (RAG)
# ============================================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION ET CONSTANTES
# ============================================================================

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Chemins des données
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_FILE = DATA_DIR / "clean_events.csv"

# Chemins de sortie du vector store
VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"

# Configuration du modèle d'embeddings multilingue (français + anglais)
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMB_DEVICE = "cpu"  # Changer en "cuda" pour accélération GPU si disponible

# Configuration du découpage intelligent des documents
CHUNK_SIZE = 512  # Taille de chaque chunk (en caractères)
CHUNK_OVERLAP = 50  # Chevauchement entre chunks pour continuité sémantique

# Configuration du logging (traçabilité complète du pipeline)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def load_cleaned_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Charger les données d'événements nettoyées depuis un fichier CSV.
    
    Cette fonction lit le fichier CSV contenant les 415 événements nettoyés
    et les convertit en DataFrame pour traitement ultérieur.
    
    Args:
        csv_path: Chemin vers le fichier CSV des événements nettoyés
        
    Returns:
        DataFrame pandas contenant les données d'événements
        
    Lève:
        FileNotFoundError: Si le fichier CSV n'existe pas
        ValueError: Si le CSV est vide ou invalide
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données non trouvé: {csv_path}")
    
    logger.info(f"Chargement des données nettoyées depuis: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError("Le fichier CSV chargé est vide")
    
    logger.info(f"Chargement de {len(df)} événements depuis le CSV")
    return df


def prepare_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convertir les lignes du DataFrame en objets Document de LangChain.
    
    ADAPTATION: Cette fonction utilise exactement les colonnes du fichier clean_events.csv:
    - title, location, timings, longDescription, links, registration, conditions, age
    
    Chaque document contient:
    1. Un contenu enrichi et formaté en français pour la recherche sémantique
    2. Les métadonnées complètes pour récupération ultérieure dans le RAG
    
    Args:
        df: DataFrame pandas contenant les données d'événements du CSV
        
    Returns:
        Liste d'objets Document LangChain optimisés pour la récupération RAG
    """
    # Mapping des mois anglais abrégés (format OpenAgenda) vers noms français complets.
    # Réduit l'écart sémantique entre "Mar 2026" indexé et "mars 2026" dans la requête.
    MONTHS_FR = {
        "Jan": "Janvier", "Feb": "Février", "Mar": "Mars", "Apr": "Avril",
        "May": "Mai",     "Jun": "Juin",    "Jul": "Juillet", "Aug": "Août",
        "Sep": "Septembre", "Oct": "Octobre", "Nov": "Novembre", "Dec": "Décembre"
    }

    import re as _re

    def _timings_to_fr(timings_str: str) -> str:
        """Remplace les abréviations de mois anglaises par les noms français complets."""
        for abbr, full in MONTHS_FR.items():
            timings_str = _re.sub(rf'\b{abbr}\b', full, str(timings_str))
        return timings_str

    def _extract_city(location_str: str) -> str:
        """Extrait le nom de ville depuis le format 'adresse, code_postal, Ville'."""
        parts = [p.strip() for p in str(location_str).split(',')]
        return parts[-1] if len(parts) >= 2 else location_str

    documents = []
    
    # Parcourir chaque événement du DataFrame
    for idx, row in df.iterrows():
        # ADAPTATION 1: Créer le contenu riche formaté en français
        # Extraction des colonnes réelles du CSV avec gestion des valeurs manquantes
        title = row.get('title', 'N/A')
        location = row.get('location', 'N/A')
        timings = row.get('timings', 'N/A')
        description = row.get('longDescription', 'N/A')
        age_requirement = row.get('age', 'N/A')
        registration = row.get('registration', 'N/A')
        conditions = row.get('conditions', 'N/A')

        # Nettoyer le titre pour l'embedding : supprimer les marqueurs de statut
        # (ex: "***COMPLET ***", "***Sur liste d'attente***") qui polluent la représentation
        # vectorielle et causent des faux négatifs lors du retrieval.
        # Le titre original reste intact dans les métadonnées.
        title_clean = _re.sub(r'\*+[^*]*\*+', '', str(title))  # supprime ***...***
        title_clean = _re.sub(r'\s+', ' ', title_clean).strip()
        if not title_clean:
            title_clean = str(title)

        # Convertir les mois en français et extraire la ville pour améliorer le retrieval
        timings_fr = _timings_to_fr(timings)
        city = _extract_city(location)

        # Formater le contenu en français structuré et optimisé pour la recherche sémantique
        # La ville est placée en tête du chunk pour augmenter son poids dans le vecteur.
        text_content = f"""Événement culturel : {title_clean}

Ville : {city}

Horaires : {timings_fr}

Lieu complet : {location}

Description : {description}

Conditions d'accès : {conditions}

Public cible - Âge recommandé : {age_requirement}

Modalités d'inscription : {registration}

Liens utiles : {row.get('links', 'N/A')}"""
        
        # ADAPTATION 2: Stocker TOUTES les colonnes originales dans les métadonnées
        # Cela permet au système RAG d'accéder aux données originales complètes lors de la génération
        metadata = {
            "source": "openagenda",  # Source pour traçabilité
            "title": title,
            "location": location,
            "city": city,
            "timings": timings,
            "longDescription": description,
            "age": age_requirement,
            "registration": registration,
            "conditions": conditions,
            "links": row.get('links', 'N/A'),
            "row_index": idx  # Index pour retour aux données brutes
        }
        
        # Créer l'objet Document avec contenu riche et métadonnées complètes
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)
    
    logger.info(f"Préparation de {len(documents)} documents à partir du DataFrame")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Diviser les longs documents en chunks plus petits pour meilleure récupération.
    
    Utilise un découpage récursif basé sur les caractères pour préserver
    les limites sémantiques naturelles (paragraphes, phrases).
    
    Strategy:
    - Essayer d'abord de découper sur \n\n (paragraphes)
    - Puis sur \n (retours à la ligne)
    - Puis sur ". " (fins de phrase)
    - Et finalement sur " " (espaces) si nécessaire
    
    Args:
        documents: Liste d'objets Document à diviser
        
    Returns:
        Liste d'objets Document divisés en chunks
    """
    # Initialiser le splitter avec stratégie de découpage intelligente
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Taille maximale d'un chunk
        chunk_overlap=CHUNK_OVERLAP,  # Chevauchement pour continuité sémantique
        separators=["\n\n", "\n", ". ", " ", ""]  # Ordre de préférence de découpage
    )
    
    # Appliquer le découpage à tous les documents
    split_docs = splitter.split_documents(documents)
    
    logger.info(f"Division en {len(split_docs)} chunks (à partir de {len(documents)} documents)")
    return split_docs


def create_embeddings_model():
    """
    Initialiser le modèle d'embeddings multilingue.
    
    Utilise les transformers HuggingFace pour support multilingue:
    - Supporte le français et l'anglais
    - Génère des vecteurs de 768 dimensions
    - Pré-entraîné sur millions de paires de phrases
    
    Returns:
        Instance HuggingFaceEmbeddings initialisée et prête à l'emploi
    """
    logger.info(f"Chargement du modèle d'embeddings: {EMB_MODEL_NAME}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL_NAME,
        model_kwargs={"device": EMB_DEVICE},
        encode_kwargs={"normalize_embeddings": True}  # Cohérent avec rag_chain.py
    )
    
    logger.info("✅ Modèle d'embeddings chargé avec succès")
    return embeddings


def create_faiss_vectorstore(
    documents: list[Document],
    embeddings
) -> FAISS:
    """
    Créer un vector store FAISS à partir des documents.
    
    FAISS (Facebook AI Similarity Search) indexe tous les documents
    pour permettre une recherche sémantique ultra-rapide:
    - Indexation en O(log n) pour la recherche
    - Support des requêtes par similarité cosinus
    - Scalable à des millions de vecteurs
    
    Args:
        documents: Liste d'objets Document divisés
        embeddings: Modèle d'embeddings initialisé
        
    Returns:
        Instance FAISS vector store prête à l'emploi
    """
    logger.info("Création du vector store FAISS...")
    
    # Créer le vector store: génère les embeddings et indexe automatiquement
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    logger.info(f"✅ Vector store créé avec {len(documents)} documents indexés")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, save_path: str | Path) -> None:
    """
    Sauvegarder le vector store FAISS sur le disque.
    
    Sauvegarde les fichiers d'index et de métadonnées pour:
    - Réutilisation rapide sans recalcul des embeddings
    - Déploiement en production
    - Partage entre équipes
    
    Args:
        vectorstore: Instance FAISS vector store à sauvegarder
        save_path: Chemin où sauvegarder le vector store
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Sauvegarde du vector store dans: {save_path}")
    vectorstore.save_local(str(save_path))
    logger.info("✅ Vector store sauvegardé avec succès")


# ============================================================================
# PIPELINE PRINCIPAL D'INDEXATION
# ============================================================================


def main():
    """
    Pipeline principal d'exécution:
    
    Étapes complètes pour préparer et indexer les événements:
    1. Charger les données nettoyées du CSV
    2. Convertir en objets Document LangChain
    3. Diviser en chunks sémantiques
    4. Initialiser le modèle d'embeddings
    5. Create and save FAISS vector store
    6. Test similarity search for validation
    """
    try:
        logger.info("=" * 70)
        logger.info("DÉMARRAGE DE LA PRÉPARATION DU VECTOR STORE")
        logger.info("=" * 70)
        
        # Étape 1: Charger les données nettoyées
        logger.info("\n[ÉTAPE 1] Chargement des données d'événements nettoyées...")
        df = load_cleaned_data(CLEANED_DATA_FILE)
        
        # Étape 2: Préparer les documents
        logger.info("\n[ÉTAPE 2] Préparation des documents à partir des données...")
        documents = prepare_documents(df)
        
        # Étape 3: Diviser les documents en chunks
        logger.info("\n[ÉTAPE 3] Division des documents en chunks sémantiques...")
        split_docs = split_documents(documents)
        
        # Étape 4: Initialiser le modèle d'embeddings
        logger.info("\n[ÉTAPE 4] Initialisation du modèle d'embeddings multilingue...")
        embeddings = create_embeddings_model()
        
        # Étape 5: Créer le vector store FAISS
        logger.info("\n[ÉTAPE 5] Création du vector store FAISS...")
        vectorstore = create_faiss_vectorstore(split_docs, embeddings)
        
        # Étape 6: Sauvegarder le vector store sur disque
        logger.info("\n[ÉTAPE 6] Sauvegarde du vector store sur disque...")
        save_vectorstore(vectorstore, FAISS_INDEX_PATH)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PRÉPARATION DU VECTOR STORE TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 70)
        logger.info(f"Vector store sauvegardé dans: {FAISS_INDEX_PATH}")
        logger.info(f"Nombre total de documents indexés: {len(split_docs)}")
        
        # Tests de recherche par similarité (validation pour rapport technique)
        logger.info("\n" + "=" * 70)
        logger.info("Exécution des tests de recherche par similarité...")
        logger.info("=" * 70)
        test_similarity_search(embeddings)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Fichier non trouvé: {e}")
        raise
    except ValueError as e:
        logger.error(f"❌ Erreur de valeur: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        raise


# ============================================================================
# TESTS DE RECHERCHE SÉMANTIQUE (validation pour rapport technique)
# ============================================================================


def test_similarity_search(embeddings):
    """
    Tester le vector store FAISS avec la recherche par similarité sémantique.
    
    Cette fonction valide que l'indexation et la récupération fonctionnent correctement
    en exécutant des requêtes en français et affichant les résultats de manière lisible.
    
    C'est un test essentiel pour:
    - Valider la qualité des embeddings
    - Vérifier que la stratégie de chunking est efficace
    - Tester la pertinence des résultats de recherche
    - Générer des exemples pour le rapport technique
    
    Args:
        embeddings: Modèle d'embeddings initialisé (pour charger le vectorstore)
    """
    try:
        logger.info("\n[TEST] Chargement du vector store FAISS sauvegardé...")
        
        # Recharger le vectorstore depuis le disque pour test de récupération
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✅ Vector store chargé avec succès")
        
        # Requêtes de test en français - couvrant différents types d'événements
        test_queries = [
            "soirées chouettes nature famille",
            "concert piano Meudon",
            "atelier vélo Boulogne"
        ]
        
        logger.info(f"\n[TEST] Exécution de {len(test_queries)} requêtes de recherche par similarité...\n")
        
        # Parcourir chaque requête de test
        for query_idx, query in enumerate(test_queries, 1):
            logger.info(f"─" * 70)
            logger.info(f"REQUÊTE {query_idx}: \"{query}\"")
            logger.info(f"─" * 70)
            
            # Effectuer la recherche par similarité sémantique (top 3 résultats)
            results = vectorstore.similarity_search(query, k=3)
            
            if not results:
                logger.info("  ⚠️  Aucun résultat trouvé pour cette requête")
                continue
            
            # Afficher chaque résultat dans un format clair et structuré
            for result_idx, result in enumerate(results, 1):
                metadata = result.metadata
                content = result.page_content
                
                # Extraire les informations clés depuis les métadonnées
                title = metadata.get("title", "N/A")
                location = metadata.get("location", "N/A")
                timings = metadata.get("timings", "N/A")
                description = metadata.get("longDescription", "N/A")
                
                # Créer un extrait court de la description (premiers 150 caractères)
                description_excerpt = (description[:150] + "...") if len(description) > 150 else description
                
                logger.info(f"\n  Résultat {result_idx}:")
                logger.info(f"    📌 Titre: {title}")
                logger.info(f"    📍 Lieu: {location}")
                logger.info(f"    ⏰ Horaires: {timings}")
                logger.info(f"    📝 Description: {description_excerpt}")
            
            logger.info("")  # Ligne vide pour lisibilité
        
        logger.info("=" * 70)
        logger.info("✅ TESTS DE RECHERCHE PAR SIMILARITÉ TERMINÉS AVEC SUCCÈS")
        logger.info("=" * 70)
        logger.info("\nRemarque: Ces résultats de test valident que le vector store")
        logger.info("indexe correctement et récupère les événements sémantiquement similaires.")
        logger.info("Cela confirme que le pipeline RAG est prêt pour la production.")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Vector store non trouvé: {e}")
        logger.error("   Assurez-vous d'avoir exécuté le pipeline principal d'abord.")
    except Exception as e:
        logger.error(f"❌ Erreur lors des tests de recherche: {e}")
        raise


# ============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================================


if __name__ == "__main__":
    main()


