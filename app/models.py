from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class QuestionRequest(BaseModel):
    """Modèle pour les requêtes POST /ask"""
    
    question: str = Field(
        ...,
        description="La question à poser à l'assistant RAG",
        min_length=1,
        max_length=500,
        example="Quels événements culturels y a-t-il à Paris en avril?"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres optionnels pour la recherche (ex: ville, date, type d'événement)",
        example={"ville": "Paris", "date": "2026-04", "type": "concert"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quels événements culturels y a-t-il à Paris en avril?",
                "filters": {
                    "ville": "Paris",
                    "date": "2026-04",
                    "type": "concert"
                }
            }
        }


class SourceItem(BaseModel):
    """Modèle pour un élément source dans la réponse RAG"""
    
    title: str = Field(
        ...,
        description="Titre de l'événement",
        example="Festival d'Art Contemporain"
    )
    
    location: str = Field(
        ...,
        description="Lieu de l'événement",
        example="Paris, France"
    )
    
    date: str = Field(
        ...,
        description="Date de l'événement",
        example="2026-04-15"
    )


class AnswerResponse(BaseModel):
    """Modèle pour les réponses de l'API RAG"""
    
    answer: str = Field(
        ...,
        description="La réponse générada par le modèle RAG",
        example="Il y a plusieurs événements culturels à Paris en avril, notamment le Festival d'Art Contemporain..."
    )
    
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="Liste des sources/événements utilisés pour générer la réponse",
        example=[
            {
                "title": "Festival d'Art Contemporain",
                "location": "Paris, France",
                "date": "2026-04-15"
            },
            {
                "title": "Exposition Musée du Louvre",
                "location": "Paris, France",
                "date": "2026-04-20"
            }
        ]
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiance de la réponse (score entre 0 et 1)",
        example=0.87
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Il y a plusieurs événements culturels à Paris en avril, notamment le Festival d'Art Contemporain qui se déroule le 15 avril au Musée d'Art Moderne.",
                "sources": [
                    {
                        "title": "Festival d'Art Contemporain",
                        "location": "Paris, France",
                        "date": "2026-04-15"
                    },
                    {
                        "title": "Exposition Musée du Louvre",
                        "location": "Paris, France",
                        "date": "2026-04-20"
                    }
                ],
                "confidence": 0.87
            }
        }
