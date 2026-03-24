# Rapport technique — Assistant RAG Événements Culturels (Puls-Events POC)

> **Mission** : POC d'un chatbot intelligent pour la recommandation d'événements culturels, commandé par Jérémy (responsable technique Puls-Events).  
> **Stack** : Python 3.12 · FastAPI · LangChain · Mistral · FAISS · Docker

---

## 1. Objectifs du projet

**Contexte** : Puls-Events veut démontrer à ses équipes produit et marketing qu'un chatbot peut répondre en langage naturel aux questions sur les événements culturels, en s'appuyant sur ses propres données.

**Problématique** : Un LLM seul ne connaît pas un catalogue d'événements local et récent. Un système RAG (Retrieval-Augmented Generation) résout ce problème en injectant dynamiquement les données pertinentes dans le contexte de la génération.

**Objectif du POC** : démontrer la faisabilité technique (RAG fonctionnel, API exposée, conteneurisé) et la valeur métier (réponses précises et sourcées).

**Périmètre** : événements culturels de la zone **Grand Paris Seine Ouest (GPSO)** — 4 963 événements issus d'Open Agenda, période ≤ 1 an.

---

## 2. Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG                             │
│                                                                 │
│  Open Agenda API                                                │
│       │                                                         │
│       ▼                                                         │
│  [src/data/]  ──────  clean_events.csv  (4 963 événements)      │
│  Fetching + nettoyage                                           │
│       │                                                         │
│       ▼                                                         │
│  [scripts/01_prepare_vector_store.py]                           │
│  RecursiveCharacterTextSplitter (chunk=512, overlap=50)         │
│  HuggingFace Embeddings (paraphrase-multilingual-mpnet-base-v2) │
│  FAISS index  ──── rag/vectorstore/faiss_index/                 │
│       │                                                         │
│       ▼                                                         │
│  [app/rag_chain.py]  ConversationalRetrievalChain               │
│  Retriever MMR (k=6, fetch_k=20)  +  ChatMistralAI             │
│  ConversationBufferMemory (RAM, court-terme)                    │
│       │                                                         │
│       ▼                                                         │
│  [main.py]  FastAPI                                             │
│  POST /ask  ──  POST /rebuild  ──  GET /health                  │
└─────────────────────────────────────────────────────────────────┘
```

| Composant | Technologie |
|---|---|
| LLM | Mistral Large (via API `mistral-large-latest`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Vectorstore | FAISS (CPU) |
| Orchestration | LangChain + `langchain_classic` |
| API | FastAPI + Uvicorn |
| Conteneurisation | Docker multi-stage (python:3.12-slim + uv) |

---

## 3. Préparation et vectorisation des données

**Source** : API Open Agenda, filtrée sur la zone GPSO, période ≤ 1 an.

**Pipeline** (`src/data/`) :
1. `01_discover_agendas.py` — découverte des agendas GPSO
2. `02_openagenda_fetcher.py` — récupération des événements bruts → `data/raw_events.json`
3. `03_clean_data.py` — nettoyage → `data/clean_events.csv`

**Anomalies corrigées** : événements sans description, doublons, dates inférieures à -1 an, encodage HTML nettoyé.

**Colonnes retenues** : `title`, `location`, `timings`, `longDescription`, `links`, `registration`, `conditions`, `age`.

**Chunking** (`scripts/01_prepare_vector_store.py`) :
- `RecursiveCharacterTextSplitter` : taille = **512 caractères**, chevauchement = **50**.
- Choix : 512 caractères offre un bon compromis entre précision sémantique et contexte suffisant pour un événement court.

**Embeddings** :
- Modèle : `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- Multilingue (fr/en), exécuté en local sur CPU, vecteurs normalisés.
- Cohérence garantie : même modèle à l'indexation et à la requête.

---

## 4. Choix du modèle NLP

**Modèle** : `mistral-large-latest` via l'API Mistral AI.

**Pourquoi Mistral ?**
- Excellente qualité en français (langue principale des données).
- API simple, compatible LangChain nativement (`langchain_mistralai`).
- Coût modéré pour un POC vs GPT-4.
- `temperature=0.3` : réponses précises et peu hallucinées.

**Prompt système** :
```
Tu es un assistant expert en événements culturels pour Grand Paris Seine Ouest.
Réponds uniquement en français, de manière claire et pédagogique.
Utilise uniquement le contexte suivant pour répondre. Si tu ne sais pas, dis-le.
```

**Mémoire court-terme** : `ConversationBufferMemory` (RAM) — stocke l'historique de la session pour permettre des questions de suivi (ex : "et à Sèvres ?"). Réinitialisée à chaque redémarrage (suffisant pour le POC).

**Limites** : dépendance à l'API Mistral (latence réseau, quota), pas de mémoire longue durée, max 2 048 tokens en sortie.

---

## 5. Construction de la base vectorielle

**Index FAISS** (`IndexFlatL2` via LangChain) :
- Stocké dans `rag/vectorstore/faiss_index/` (fichiers `index.faiss` + `index.pkl`).
- Recherche par **MMR** (Maximal Marginal Relevance) : `k=6` résultats finaux, `fetch_k=20` candidats initiaux, `lambda_mult=0.7` — favorise la diversité géographique (évite 6 chunks du même lieu).

**Métadonnées conservées par chunk** : `title`, `location`, `timings` — injectées dans la réponse JSON (`sources`).

**Reconstruction** : appel à `POST /rebuild` (ou exécution directe de `scripts/01_prepare_vector_store.py`).

---

## 6. API et endpoints exposés

**Framework** : FastAPI — choisi pour la rapidité de développement, la validation automatique Pydantic et la doc Swagger intégrée.

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Interface de chat HTML |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Question → réponse RAG |
| `POST` | `/rebuild` | Reconstruction de l'index (background task) |
| `GET` | `/docs` | Documentation Swagger |

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts y a-t-il à Boulogne en avril ?", "filters": {}}'
```

**Réponse** :
```json
{
  "answer": "Voici les concerts disponibles à Boulogne-Billancourt en avril...",
  "sources": [
    {"title": "Concert Jazz", "location": "Boulogne-Billancourt", "date": "2026-04-12"}
  ],
  "confidence": 0.82
}
```

### POST /rebuild

Lance `scripts/01_prepare_vector_store.py` en **background task** (HTTP 202 immédiat). Utile pour mettre à jour l'index après récupération de nouvelles données.

**Gestion des erreurs** : toute exception RAG est capturée et retournée avec `confidence: 0.0` plutôt qu'un HTTP 500, pour ne pas bloquer le front.

---

## 7. Évaluation du système

### Jeu de test annoté (extrait)

| # | Question | Réponse de référence | Score observé |
|---|---|---|---|
| 1 | "Quels événements y a-t-il à Paris en mars 2026 ?" | Liste d'événements GPSO en mars | ✅ Pertinent |
| 2 | "Quel type d'événements sont disponibles ?" | Catégories : concerts, balades, expositions… | ✅ Pertinent |
| 3 | "Quels sont les événements à Boulogne ?" | Events localisés à Boulogne-Billancourt | ✅ Pertinent |
| 4 | "Y a-t-il des événements pour enfants ?" | Filtrage par condition d'âge | ✅ Pertinent |
| 5 | "Quels événements se passent ce week-end à Meudon ?" | Événements géolocalisés + datés | ⚠️ Partiel (dépend de la date) |

**Nombre d'exemples** : 5 questions couvertes dans les tests unitaires automatisés.

**Métriques** :
- **Couverture des sources** : le retriever MMR retourne 6 chunks diversifiés géographiquement.
- **Score de confiance** : calculé comme la moyenne des scores de similarité cosinus des sources retournées (0–1).
- **Qualité subjective** : réponses en français, sourcées, sans hallucination sur les données présentes dans l'index.

**Tests unitaires** (`tests/test_api.py`) — 9 tests couvrant :
- `GET /health` : status 200 + format
- `POST /ask` : status 200, format de réponse (answer/sources/confidence), filtres, question vide
- `POST /rebuild` : status 202, format de réponse

```bash
pytest tests/ -v
```

---

## 8. Recommandations et perspectives

**Ce qui fonctionne bien** :
- Pipeline RAG complet et dockerisé, opérationnel en 1 commande.
- Diversité géographique assurée par le retriever MMR.
- Mémoire de session pour les questions de suivi.

**Limites du POC** :
- Index FAISS rechargé en mémoire au démarrage (latence ~10–30 s selon la machine).
- `ConversationBufferMemory` non persistante : la mémoire est perdue à chaque redémarrage.
- Données statiques : l'index ne se met pas à jour automatiquement.
- Coût API Mistral à surveiller en production.

**Améliorations possibles** :
- Ajout d'un scheduler (Celery / APScheduler) pour la mise à jour automatique de l'index.
- Remplacement de `ConversationBufferMemory` par `ConversationSummaryMemory` pour les longues sessions.
- Passage à un index FAISS IVF pour de meilleures performances sur de gros volumes.
- Mise en place de métriques RAG automatisées (RAGAS : faithfulness, answer relevancy).
- Passage en production via Kubernetes ou un service serverless avec FAISS sur S3.

---

## 9. Organisation du dépôt GitHub

```
OC_P7/
├── app/                        # Logique applicative
│   ├── models.py               # Schémas Pydantic (QuestionRequest, AnswerResponse)
│   ├── rag_chain.py            # Initialisation FAISS + LLM + chaîne RAG
│   └── __init__.py
├── data/
│   ├── clean_events.csv        # 4 963 événements nettoyés (indexés)
│   └── raw_events.json         # Données brutes OpenAgenda
├── projet/                     # Documents de mission (hors code)
├── rag/
│   └── vectorstore/
│       └── faiss_index/        # Index FAISS persisté (index.faiss + index.pkl)
├── scripts/
│   ├── 01_prepare_vector_store.py  # Chunking + embeddings + FAISS
│   └── 02_rag_pipeline.py          # Pipeline RAG standalone (hors API)
├── src/data/                   # Scripts de collecte et nettoyage
│   ├── 01_discover_agendas.py
│   ├── 02_openagenda_fetcher.py
│   └── 03_clean_data.py
├── static/
│   └── index.html              # Interface de chat HTML intégrée
├── tests/
│   ├── conftest.py             # Fixtures pytest (async_client)
│   └── test_api.py             # 9 tests unitaires (health, /ask, /rebuild)
├── Dockerfile                  # Build multi-stage (python:3.12-slim + uv)
├── main.py                     # Point d'entrée FastAPI
├── pyproject.toml              # Dépendances (uv)
└── requirements.txt            # Export figé pour Docker
```

---

## 10. Annexes

### Exemple de réponse JSON complète

```json
{
  "answer": "Voici quelques événements culturels disponibles dans la zone Grand Paris Seine Ouest :\n\n1. **Soirées Chouettes** — Ville-d'Avray, 13 mars 2026 19h30. Balade nocturne guidée pour observer la faune nocturne.\n2. **Exposition Contemporaine** — Boulogne-Billancourt, avril 2026.\n\nN'hésitez pas à préciser une ville ou une date pour affiner les résultats.",
  "sources": [
    {
      "title": "Les soirées chouettes de la Maison de la Nature",
      "location": "Ville-d'Avray, 92410",
      "date": "13 Mar 2026 19h30 - 21h00"
    }
  ],
  "confidence": 0.79
}
```

### Prompt complet injecté dans la chaîne

```
System: Tu es un assistant expert en événements culturels pour Grand Paris Seine Ouest.
        Réponds uniquement en français, de manière claire et pédagogique.
        Utilise uniquement le contexte suivant pour répondre. Si tu ne sais pas, dis-le.

Human: Contexte : {context}
Question : {question}
```

### Extrait de log de démarrage

```
✅ Embeddings HuggingFace initialisés avec succès
✅ Index FAISS chargé avec succès depuis rag/vectorstore/faiss_index
✅ Retriever MMR configuré (k=6, fetch_k=20) pour diversité géographique
✅ Modèle ChatMistralAI initialisé (mistral-large-latest)
✅ Prompt conversationnel en français configuré
✅ Mémoire ConversationBufferMemory initialisée en RAM
✅ Chaîne RAG conversationnelle créée avec succès
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Variables d'environnement requises

Créer un fichier `.env` à la racine (non versionné) :

```
MISTRAL_API_KEY=your_key_here
```

---

## 🚀 Lancement en 1 commande (Docker)

```bash
docker build --no-cache -t rag-api:latest .
```

L'API est accessible sur **http://localhost:8000**  
Interface de chat : **http://localhost:8000/**  
Documentation Swagger : **http://localhost:8000/docs**

> Prérequis : fichier `.env` avec `MISTRAL_API_KEY` à la racine.  
> L'index FAISS est déjà inclus dans l'image (`rag/vectorstore/faiss_index/`).  
> Pour reconstruire l'index : `POST http://localhost:8000/rebuild`
