import argparse
import json
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import RequestException, Timeout
from tqdm import tqdm

# Constantes
AGENDA_UID = 85121895  # Grand Paris Seine Ouest - GPSO
DATA_DIR = "data"
RAW_FILE = f"{DATA_DIR}/raw_events.json"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5

os.makedirs(DATA_DIR, exist_ok=True)


def _request_with_retry(url: str, params: dict, headers: dict = None) -> Response:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if headers is None:
                headers = {}
            
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)

            status_code = response.status_code

            if status_code == 429:
                print("⚠️ Rate limit détecté (429), pause de 1s puis retry...")
                time.sleep(1)
                continue

            if 500 <= status_code < 600:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Erreur API HTTP {status_code} après {MAX_RETRIES} tentatives."
                    )
                print(
                    f"⚠️ Erreur serveur API ({status_code}) "
                    f"(tentative {attempt}/{MAX_RETRIES}), retry dans 1s..."
                )
                time.sleep(1)
                continue

            if 400 <= status_code < 500:
                error_msg = f"Erreur API HTTP {status_code}"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    pass
                raise RuntimeError(error_msg)

            response.raise_for_status()
            return response

        except Timeout:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"⚠️ Timeout API (tentative {attempt}/{MAX_RETRIES}), retry dans 1s..."
            )
            time.sleep(1)
        except RequestException:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Échec réseau/API après plusieurs tentatives."
                )
            print(
                f"⚠️ Erreur réseau/API (tentative {attempt}/{MAX_RETRIES}), retry dans 1s..."
            )
            time.sleep(1)

    raise RuntimeError("Échec des requêtes API après plusieurs tentatives.")


def _summarize_fields(events: list[dict]) -> None:
    """Affiche un résumé des champs présents dans les événements."""
    total = len(events)

    print("\n🧾 Résumé des événements récupérés")
    print(f"- Totle: {total}")

    if total == 0:
        return

    # Analyser tous les champs présents dans le premier événement
    if events:
        first_event = events[0]
        print(f"- Champs présents: {', '.join(first_event.keys())}")


def fetch_all_events(force: bool = False):
    """
    Récupère tous les événements de l'agenda GPSO avec filtres date + pagination.
    Si le fichier existe déjà et que force=False, skip le fetch.
    """
    if os.path.exists(RAW_FILE) and not force:
        print(
            f"ℹ️ {RAW_FILE} existe déjà. Utilise --force pour re-fetch les données."
        )
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            existing_events = json.load(f)
        _summarize_fields(existing_events)
        return existing_events

    load_dotenv()
    api_key = os.getenv("OPENAGENDA_API_KEY")
    if not api_key:
        raise ValueError("OPENAGENDA_API_KEY est manquante dans l'environnement.")

    one_year_ago = (datetime.now() - timedelta(days=365)).isoformat()

    url = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"
    params_base = {
        "detailed": 1,
        "size": 100,
        "timings[gte]": one_year_ago,
        "timings[lte]": "2100-01-01",
    }
    
    # Passer la clé en entête HTTP plutôt qu'en query parameter
    headers = {
        "key": api_key
    }

    all_events = []
    offset = 0
    total = None

    print("🚀 Récupération des événements GPSO en cours...")

    with tqdm(desc="Pages API", unit="page") as pbar:
        while True:
            params = params_base.copy()
            params["offset"] = offset

            response = _request_with_retry(url, params, headers)
            data = response.json()

            if total is None:
                total = data.get("total", 0)
                print(f"📊 Total attendu : {total} événements")

            events = data.get("events", [])
            all_events.extend(events)

            pbar.update(1)
            pbar.set_postfix({"récupérés": len(all_events), "total": total})

            if len(events) < params["size"]:
                break

            offset += params["size"]

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(all_events)} événements sauvegardés dans {RAW_FILE}")
    _summarize_fields(all_events)
    return all_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch des événements OpenAgenda")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le re-fetch même si le fichier existe déjà",
    )
    args = parser.parse_args()
    fetch_all_events(force=args.force)


if __name__ == "__main__":
    main()