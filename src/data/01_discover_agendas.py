import os
import re
import unicodedata
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


def _normalize_text(value: str | None):
    if not value:
        return ""
    lowered = value.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _build_area_keywords(search_terms: list[str]):
    keywords = {_normalize_text(term) for term in search_terms}
    aliases = {
        "isere": ["isere", "nord-isere", "grenoble", "grenoblois", "vienne", "viennois", "voiron", "bourgoin", "crolles", "meylan", "echirolles", "saint-martin-d-heres"],
        "savoie": ["savoie", "savoy", "chambery", "albertville"],
        "haute-savoie": ["haute-savoie", "haute savoie", "annecy", "mont-blanc"],
        "drome": ["drome", "valence", "valence-romans", "romans"],
        "ain": ["bourg-en-bresse"],
        "rhone": ["rhone", "lyon", "lyonnais", "villeurbanne"],
        "ardeche": ["ardeche", "ardechois", "privas"],
        "hautes-alpes": ["hautes-alpes", "hautes alpes", "gap"],
    }

    expanded = set(keywords)
    for key, values in aliases.items():
        if key in keywords:
            expanded.update(_normalize_text(v) for v in values)
    return sorted(expanded)


def _is_in_target_area(agenda: dict, area_keywords: list[str]):
    # Matching générique sur titre + description + slug
    title = _normalize_text(agenda.get("title"))
    description = _normalize_text(agenda.get("description"))
    slug = _normalize_text(agenda.get("slug"))
    combined = f"{title} {description} {slug}"
    tokens = set(re.findall(r"[a-z0-9]+", combined))

    for keyword in area_keywords:
        if not keyword:
            continue
        if " " in keyword or "-" in keyword:
            if keyword in combined:
                return True
        else:
            if keyword in tokens:
                return True

    return False


def _extract_events_count(agenda: dict):
    """Récupère un nombre d'événements si la donnée est disponible."""
    direct_keys = ("eventsCount", "events_count", "totalEvents", "total_events")
    for key in direct_keys:
        value = agenda.get(key)
        if isinstance(value, int):
            return value

    stats = agenda.get("stats")
    if isinstance(stats, dict):
        for key in ("events", "eventsCount", "count"):
            value = stats.get(key)
            if isinstance(value, int):
                return value

    return None


def _parse_iso_datetime(value: str | None):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_since_update(updated_at: str | None):
    parsed = _parse_iso_datetime(updated_at)
    if parsed is None:
        return None
    now = datetime.now(timezone.utc)
    return max((now - parsed).days, 0)


def _compute_relevance_score(agenda: dict, target_keywords: list[str] | None = None):
    """
    Score pour privilégier les agendas généralistes sur la zone ciblée.
    """
    title = (agenda.get("title") or "").lower()
    description = (agenda.get("description") or "").lower()
    text = f"{title} {description}"

    score = 0
    search_keywords = [keyword.lower() for keyword in (target_keywords or [])]
    if not search_keywords:
        search_keywords = ["paris"]

    score += sum(3 for keyword in search_keywords if keyword in text)

    broad_positive_keywords = [
        "culture",
        "événement",
        "evenement",
        "sortir",
        "agenda",
        "ville",
        "métropole",
        "metropole",
    ]
    narrow_negative_keywords = [
        "université",
        "universite",
        "faculté",
        "faculte",
        "diocèse",
        "diocese",
        "paroisse",
        "librairie",
        "fraternité",
        "fraternite",
        "église",
        "eglise",
        "club",
    ]

    score += sum(2 for keyword in broad_positive_keywords if keyword in text)
    score -= sum(3 for keyword in narrow_negative_keywords if keyword in text)

    total_events = agenda.get("_events_total")
    if isinstance(total_events, int):
        if total_events >= 10000:
            score += 8
        elif total_events >= 3000:
            score += 6
        elif total_events >= 1000:
            score += 4
        elif total_events >= 300:
            score += 2

    days_since_update = agenda.get("_days_since_update")
    if isinstance(days_since_update, int):
        if days_since_update <= 3:
            score += 5
        elif days_since_update <= 14:
            score += 3
        elif days_since_update <= 30:
            score += 1
        elif days_since_update > 120:
            score -= 3

    return score


def recommend_generalist_agendas(
    agendas: list[dict],
    min_events: int = 100,
    max_days_since_update: int = 30,
    top_n: int = 5,
):
    """Filtre les agendas trop spécialisés et retourne un top généraliste."""
    excluded_keywords = [
        "université",
        "universite",
        "faculté",
        "faculte",
        "diocèse",
        "diocese",
        "paroisse",
        "fraternité",
        "fraternite",
        "eglise",
        "église",
        "club",
    ]

    filtered = []
    for agenda in agendas:
        if not agenda.get("_matches_target_area", False):
            continue

        title = (agenda.get("title") or "").lower()
        description = (agenda.get("description") or "").lower()
        combined_text = f"{title} {description}"

        if any(keyword in combined_text for keyword in excluded_keywords):
            continue

        events_total = agenda.get("_events_total")
        if isinstance(events_total, int) and events_total < min_events:
            continue

        days_since_update = agenda.get("_days_since_update")
        if isinstance(days_since_update, int) and days_since_update > max_days_since_update:
            continue

        filtered.append(agenda)

    filtered.sort(
        key=lambda item: (
            -item.get("_relevance_score", 0),
            -(item.get("_events_total") or 0),
            item.get("_days_since_update")
            if item.get("_days_since_update") is not None
            else 10**9,
        )
    )

    return filtered[:top_n]


def list_agendas_for_search_terms(search_terms: list[str], size: int = 20, timeout: int = 10):
    """
    Liste les agendas OpenAgenda pour une ou plusieurs zones de recherche.
    Trie par nombre d'événements si l'information est disponible.
    """
    if not search_terms:
        raise RuntimeError("Aucun terme de recherche fourni.")

    load_dotenv()
    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        raise RuntimeError("La variable d'environnement OPENAGENDA_API_KEY est absente.")

    url = "https://api.openagenda.com/v2/agendas"
    agendas_by_uid = {}
    for term in search_terms:
        params = {
            "key": api_key,
            "search": term,
            "size": size,
            "fields": "uid,title,description,location,slug",
        }

        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in (401, 403):
                raise RuntimeError("Clé API OpenAgenda invalide (401/403). Vérifie OPENAGENDA_API_KEY.")
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Timeout après {timeout}s lors de l'appel OpenAgenda. Réessaie ou augmente le timeout."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Erreur réseau/API OpenAgenda: {exc}") from exc

        data = response.json()
        agendas = data.get("agendas", [])
        for agenda in agendas:
            uid = agenda.get("uid")
            if uid is not None and uid not in agendas_by_uid:
                agendas_by_uid[uid] = agenda

    agendas = list(agendas_by_uid.values())

    area_keywords = _build_area_keywords(search_terms)

    enriched_agendas = []
    for agenda in agendas:
        agenda_copy = dict(agenda)
        uid = agenda.get("uid")

        events_total = _extract_events_count(agenda)
        updated_at = None

        if uid:
            try:
                detail_response = requests.get(
                    f"https://api.openagenda.com/v2/agendas/{uid}",
                    params={"key": api_key},
                    timeout=timeout,
                )
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                if isinstance(detail_data, dict):
                    updated_at = detail_data.get("updatedAt")
            except requests.RequestException:
                updated_at = None

            try:
                events_response = requests.get(
                    f"https://api.openagenda.com/v2/agendas/{uid}/events",
                    params={"key": api_key, "size": 1},
                    timeout=timeout,
                )
                events_response.raise_for_status()
                events_data = events_response.json()
                if isinstance(events_data, dict):
                    total = events_data.get("total")
                    if isinstance(total, int):
                        events_total = total
            except requests.RequestException:
                pass

        agenda_copy["_events_total"] = events_total
        agenda_copy["_updated_at"] = updated_at
        agenda_copy["_days_since_update"] = _days_since_update(updated_at)
        agenda_copy["_matches_target_area"] = _is_in_target_area(agenda_copy, area_keywords)
        agenda_copy["_relevance_score"] = _compute_relevance_score(agenda_copy, target_keywords=search_terms)
        if agenda_copy["_matches_target_area"]:
            agenda_copy["_relevance_score"] += 6
        enriched_agendas.append(agenda_copy)

    enriched_agendas.sort(
        key=lambda item: (
            -item.get("_relevance_score", 0),
            -(item.get("_events_total") or 0),
            item.get("_days_since_update")
            if item.get("_days_since_update") is not None
            else 10**9,
        )
    )

    zone_label = ", ".join(search_terms)
    print(f"✅ {len(enriched_agendas)} agendas trouvés pour '{zone_label}' :")
    for agenda in enriched_agendas:
        uid = agenda.get("uid", "N/A")
        title = agenda.get("title", "Sans titre")
        city = agenda.get("location", {}).get("city", "Inconnu")
        slug = agenda.get("slug")
        events_total = agenda.get("_events_total")
        events_label = events_total if events_total is not None else "indisponible"
        days_since_update = agenda.get("_days_since_update")
        freshness_label = (
            f"il y a {days_since_update} jour(s)"
            if days_since_update is not None
            else "indisponible"
        )
        agenda_link = f"https://openagenda.com/{slug}" if slug else "indisponible"
        relevance_score = agenda.get("_relevance_score", 0)
        in_area = "oui" if agenda.get("_matches_target_area") else "non"

        print(
            "   • "
            f"UID: {uid} | "
            f"Titre: {title} | "
            f"Ville: {city} | "
            f"Événements: {events_label} | "
            f"Maj: {freshness_label} | "
            f"Zone: {in_area} | "
            f"Score: {relevance_score} | "
            f"Lien: {agenda_link}"
        )

    return enriched_agendas


def list_agendas_in_paris(size: int = 10, timeout: int = 10):
    return list_agendas_for_search_terms(search_terms=["Paris"], size=size, timeout=timeout)


if __name__ == "__main__":
    try:
        agendas = list_agendas_in_paris(size=20)
        print("\n🎯 Recommandation (généraliste Paris, actif, non spécialisé):")
        shortlisted = recommend_generalist_agendas(agendas, min_events=100, max_days_since_update=45, top_n=5)
        if not shortlisted:
            print("   Aucun agenda ne correspond aux critères sur cet échantillon.")
        else:
            for agenda in shortlisted:
                uid = agenda.get("uid", "N/A")
                title = agenda.get("title", "Sans titre")
                total = agenda.get("_events_total", "indisponible")
                freshness = agenda.get("_days_since_update")
                freshness_text = (
                    f"il y a {freshness} jour(s)"
                    if freshness is not None
                    else "indisponible"
                )
                slug = agenda.get("slug")
                link = f"https://openagenda.com/{slug}" if slug else "indisponible"
                print(
                    f"   • UID: {uid} | {title} | Événements: {total} | Maj: {freshness_text} | Lien: {link}"
                )
    except RuntimeError as error:
        print(f"❌ {error}")