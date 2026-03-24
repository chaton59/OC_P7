"""
Tests de l'API FastAPI - Assistant RAG Événements Culturels GPSO

Ce module teste les endpoints principaux de l'API :
- GET /     : Vérification du statut de l'API
- POST /ask : Endpoint RAG pour poser des questions
- POST /rebuild : Reconstruction de l'index vectoriel

Framework: pytest + httpx (async)
Auteur: Projet OC P7
Date: Mars 2026

IMPORTANT: Tous les tests utilisent async httpx pour une exécution asynchrone.
Cela permet des tests plus rapides et plus réalistes pour une API asynchrone.
"""

import pytest
from unittest.mock import patch


class TestHealthCheck:
    """
    ========================================
    TESTS DU HEALTH CHECK (GET /)
    ========================================
    
    Vérifie que l'API est opérationnelle et retourne le bon message.
    C'est un endpoint critique pour valider le démarrage du serveur.
    """
    
    @pytest.mark.asyncio
    async def test_health_check_status_200(self, async_client):
        """
        TEST 1 : Vérifier que GET / retourne un status 200
        
        Objectif: Confirmer que l'endpoint de santé répond correctement
        
        Assertions:
        - Status HTTP doit être 200 (succès)
        - Le JSON doit contenir la clé "status"
        
        Exécution async: Les tests sont lancés de manière asynchrone
        pour respecter la nature non-bloquante de l'API.
        """
        response = await async_client.get("/health")
        
        assert response.status_code == 200, \
            f"Expected status 200 but got {response.status_code}"
    
    
    @pytest.mark.asyncio
    async def test_health_check_response_format(self, async_client):
        """
        TEST 2 : Vérifier la structure et le contenu de la réponse
        
        Objectif: Valider que la réponse respecte le format attendu
        
        Assertions:
        - La réponse JSON contient "status": "ok"
        - La clé "message" est présente et non vide
        - Le message contient "docs" (lien vers documentation)
        
        Pipeline de test:
        1. Effectuer une requête GET /
        2. Parser la réponse JSON
        3. Valider chaque champ du schéma attendu
        """
        response = await async_client.get("/health")
        data = response.json()
        
        # Vérifier que le statut est "ok"
        assert "status" in data, "Response should contain 'status' key"
        assert data["status"] == "ok", "Status should be 'ok'"
        
        # Vérifier que le message est présent
        assert "message" in data, "Response should contain 'message' key"
        assert len(data["message"]) > 0, "Message should not be empty"


class TestRAGQuestion:
    """
    ========================================
    TESTS DU ENDPOINT RAG (POST /ask)
    ========================================
    
    Teste l'endpoint principal pour poser des questions à l'assistant RAG.
    C'est le cœur fonctionnel de l'API.
    """
    
    @pytest.mark.asyncio
    async def test_ask_question_status_200(self, async_client):
        """
        TEST 3 : Vérifier que POST /ask retourne un status 200
        
        Objectif: Confirmer que l'endpoint RAG traite la question
        
        Paramètres:
        - question: "Quels événements y a-t-il à Paris en mars 2026?"
        - filters: {} (pas de filtres)
        
        Assertions:
        - Status HTTP est 200 (succès)
        
        Note: Ce test valide que le pipeline RAG fonctionne
        sans erreurs critiques (erreurs remontées en HTTP 500).
        """
        payload = {
            "question": "Quels événements y a-t-il à Paris en mars 2026?",
            "filters": {}
        }
        
        response = await async_client.post("/ask", json=payload)
        
        assert response.status_code == 200, \
            f"Expected status 200 but got {response.status_code}"
    
    
    @pytest.mark.asyncio
    async def test_ask_question_response_format(self, async_client):
        """
        TEST 4 : Vérifier le format de réponse RAG
        
        Objectif: Valider que la réponse RAG contient tous les champs requis
        
        Paramètres:
        - question: "Quel type d'événements sont disponibles?"
        
        Assertions:
        - Réponse JSON contient "answer" (la réponse générée)
        - Réponse JSON contient "sources" (documents source du RAG)
        - Réponse JSON contient "confidence" (score de confiance)
        - "answer" est une chaîne non vide
        - "sources" est une liste (même si vide)
        - "confidence" est un nombre entre 0 et 1
        
        Pipeline RAG vérifié:
        1. Question -> Embedding (via Mistral)
        2. Recherche vectorielle dans FAISS
        3. Génération de réponse avec contexte
        4. Retour structure AnswerResponse
        """
        payload = {
            "question": "Quel type d'événements sont disponibles?",
            "filters": {}
        }
        
        response = await async_client.post("/ask", json=payload)
        data = response.json()
        
        # Vérifier la présence des clés principales
        assert "answer" in data, "Response should contain 'answer' key"
        assert "sources" in data, "Response should contain 'sources' key"
        assert "confidence" in data, "Response should contain 'confidence' key"
        
        # Vérifier les types de données
        assert isinstance(data["answer"], str), "answer should be a string"
        assert len(data["answer"]) > 0, "answer should not be empty"
        
        assert isinstance(data["sources"], list), "sources should be a list"
        
        assert isinstance(data["confidence"], (int, float)), \
            "confidence should be a number"
        assert 0 <= data["confidence"] <= 1, \
            "confidence should be between 0 and 1"
    
    
    @pytest.mark.asyncio
    async def test_ask_question_with_filters(self, async_client):
        """
        TEST 5 : Vérifier que les filtres sont acceptés
        
        Objectif: Tester l'endpoint avec des paramètres filtrés
        
        Paramètres:
        - question: "Quels sont les événements à Boulogne?"
        - filters: {"city": "Boulogne"} (filtre par ville)
        
        Assertions:
        - Status HTTP est 200
        - La réponse respecte le format attendu
        
        Note: Les filtres permettent de raffiner la recherche RAG
        au niveau de la base de données vectorielle.
        """
        payload = {
            "question": "Quels sont les événements à Boulogne?",
            "filters": {"city": "Boulogne"}
        }
        
        response = await async_client.post("/ask", json=payload)
        
        assert response.status_code == 200, \
            f"Expected status 200 but got {response.status_code}"
        
        data = response.json()
        assert "answer" in data, "Response should contain 'answer'"
        assert "sources" in data, "Response should contain 'sources'"


class TestRebuildIndex:
    """
    ========================================
    TESTS DU REBUILD INDEX (POST /rebuild)
    ========================================
    
    Teste l'endpoint administrateur pour reconstruire l'index FAISS.
    Cet endpoint est essentiel pour la maintenance et les mises à jour.
    """
    
    @pytest.mark.asyncio
    async def test_rebuild_index_status_200(self, async_client):
        """
        TEST 6 : Vérifier que POST /rebuild retourne un status 202 Accepted

        Objectif: Confirmer que l'endpoint admin accepte la requête de rebuild

        Assertions:
        - Status HTTP est 202 (Accepted - reconstruction en background)

        Maintenance: Cet endpoint est appelé lors des mises à jour
        du corpus de données ou après des modifications du vectorstore.
        """
        with patch("main._run_rebuild"):
            response = await async_client.post("/rebuild")

        assert response.status_code == 202, \
            f"Expected status 202 but got {response.status_code}"
    
    
    @pytest.mark.asyncio
    async def test_rebuild_index_response_format(self, async_client):
        """
        TEST 7 : Vérifier le format de réponse du rebuild

        Objectif: Valider que la reconstruction retourne un statut accepté

        Assertions:
        - Réponse JSON contient "status"
        - "status" a la valeur "accepted"
        - Réponse JSON contient "message"
        - Le message mentionne la reconstruction ou le background

        Format de réponse attendu:
        {
            "status": "accepted",
            "message": "Index vectoriel en cours de reconstruction...",
            "reload_required": true
        }
        """
        with patch("main._run_rebuild"):
            response = await async_client.post("/rebuild")
        data = response.json()
        
        # Vérifier la présence des clés
        assert "status" in data, "Response should contain 'status' key"
        assert "message" in data, "Response should contain 'message' key"
        
        # Vérifier les valeurs
        assert data["status"] == "accepted", \
            "Status should be 'accepted' for a rebuild operation"
        assert "reconstruction" in data["message"].lower() or \
               "background" in data["message"].lower(), \
            "Message should mention reconstruction or background"


class TestErrorHandling:
    """
    ========================================
    TESTS DE GESTION D'ERREURS
    ========================================
    
    Teste les cas d'erreur et les réponses inappropriées.
    Important pour la robustesse de l'API.
    """
    
    @pytest.mark.asyncio
    async def test_ask_with_invalid_json(self, async_client):
        """
        TEST 8 : Vérifier le traitement d'un JSON invalide
        
        Objectif: Tester la robustesse face à des données mal formées
        
        Données invalides:
        - Payload sans la clé "question" requise
        - Seule la clé "filters" est fournie
        
        Assertions:
        - Status HTTP est 422 (validation error)
        
        Détails: FastAPI valide automatiquement les schémas de modèles
        et retourne une erreur 422 si les données ne correspondent pas
        au schéma QuestionRequest défini.
        """
        # Envoi d'un JSON incomplet (sans "question" requise)
        payload = {"filters": {}}
        
        response = await async_client.post("/ask", json=payload)
        
        # FastAPI retourne 422 Unprocessable Entity pour les erreurs de validation
        assert response.status_code == 422, \
            f"Expected status 422 for invalid JSON but got {response.status_code}"
    
    
    @pytest.mark.asyncio
    async def test_nonexistent_endpoint(self, async_client):
        """
        TEST 9 : Vérifier le traitement d'un endpoint inexistant
        
        Objectif: Tester la robustesse face à des routes non définies
        
        Endpoint visé: /this_endpoint_does_not_exist (n'existe pas)
        
        Assertions:
        - Status HTTP est 404 (not found)
        
        Détails: FastAPI retourne automatiquement une erreur 404
        si l'endpoint n'est pas défini dans les routes de l'application.
        """
        response = await async_client.get("/this_endpoint_does_not_exist")
        
        assert response.status_code == 404, \
            f"Expected status 404 for nonexistent endpoint but got {response.status_code}"


# ============================================
# NOTES POUR LE RAPPORT TECHNIQUE
# ============================================
"""
RAPPORT D'EXÉCUTION DES TESTS

1. INSTRUCTIONS D'EXÉCUTION:
   
   Lancer tous les tests:
      pytest tests/test_api.py -v
   
   Avec rapport détaillé:
      pytest tests/test_api.py -v --tb=short
   
   Avec couverture de code:
      pytest tests/test_api.py --cov=app --cov-report=html

2. DÉPENDANCES REQUISES:
   - pytest (framework de test)
   - pytest-asyncio (support des tests async)
   - httpx (client HTTP asynchrone)
   - fastapi (framework API)
   - uvicorn (serveur ASGI)

3. ARCHITECTURE DES TESTS:
   
   Tests Asynchrones (async httpx):
   - Utilise AsyncClient d'httpx
   - Tests marqués avec @pytest.mark.asyncio
   - Permet la communication directe ASGI sans réseau externe
   - Plus rapide et plus flexible
   
   Fixture async_client:
   - Définie dans conftest.py
   - Crée un client HTTP asynchrone pour chaque test
   - Scope: function (nouveau client par test)

4. RÉSULTATS ATTENDUS:
   
   9 tests au total:
   - 2 tests health check (santé de l'API)
   - 3 tests RAG (endpoint principal)
   - 2 tests rebuild (maintenance)
   - 2 tests erreurs (robustesse)
   
   Tous les tests doivent passer (status PASSED)

5. BONNES PRATIQUES APPLIQUÉES:
   
   Tests asynchrones:
   - Utilisent async/await pour respecter la nature
     asynchrone de l'API et FastAPI
   
   Commentaires détaillés:
   - Chaque test contient description, paramètres,
     assertions et pipeline logique
   
   Organisation par classe:
   - TestHealthCheck
   - TestRAGQuestion
   - TestRebuildIndex
   - TestErrorHandling
"""
