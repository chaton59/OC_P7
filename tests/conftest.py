"""
Configuration pytest pour les tests de l'API FastAPI

Ce fichier configure les fixtures communes pour tous les tests de l'API.
Utilise async httpx pour des tests asynchrones performants.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent (racine du projet) au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
import httpx
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """
    Fixture pytest pour créer un TestClient FastAPI (sync).
    
    Utilisée comme fallback pour les tests synchrones.
    
    Retour:
        TestClient: Client HTTP synchrone pour faire des requêtes
    """
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client():
    """
    Fixture pytest pour créer un client HTTP async httpx.
    
    IMPORTANT: Utiliser avec @pytest.mark.asyncio
    
    Permet de faire des requêtes HTTP asynchrones vers l'API FastAPI.
    Cette fixture crée un client ASGI qui communique directement avec
    l'application FastAPI sans passer par un réseau externe.
    
    Retour:
        httpx.AsyncClient: Client HTTP asynchrone pour les requêtes
    
    Usage:
        @pytest.mark.asyncio
        async def test_something(async_client):
            response = await async_client.get("/")
            assert response.status_code == 200
    """
    from httpx import ASGITransport
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


def pytest_configure(config):
    """
    Configuration de pytest avec markers personnalisés pour organiser les tests.
    """
    config.addinivalue_line(
        "markers", "health: tests du health check de l'API"
    )
    config.addinivalue_line(
        "markers", "rag: tests de l'endpoint RAG pour poser des questions"
    )
    config.addinivalue_line(
        "markers", "admin: tests des endpoints administrateur (rebuild, etc.)"
    )
    config.addinivalue_line(
        "markers", "error_handling: tests de gestion d'erreurs et cas invalides"
    )
