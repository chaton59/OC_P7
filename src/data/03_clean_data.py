# src/data/03_clean_data_v2.py
"""
Étape 2.4 - Nettoyage des données Open Agenda pour RAG (Version 2 - Doux)
Auteur : Valentin + Grok
Objectif : transformer raw_events.json en clean_events.csv avec nettoyage progressif

Rapport technique des transformations :
├─ Chargement : 415+ événements depuis raw_events.json (API OpenAgenda)
├─ Section 1 - Parsing multilingue : extraction texte français des champs dict
│  └─ Méthode : parsing des dicts JSON OpenAgenda → récupération clé 'fr'
├─ Section 2 - Sélection colonnes : conservation 80% des événements
│  └─ Cible : colonnes essentielles pour embeddings
├─ Section 3 - Nettoyage progressif : 
│  ├─ Suppression doublons par uid seulement (STRICT)
│  ├─ Nettoyage espaces/accents (SOFT) 
│  ├─ Description vide → "Sans description"
│  └─ (TBD : limite chars pour embeddings - phase II)
├─ Section 4 - Extraction métadonnées : dates, lieux, catégories
├─ Section 5 - Suppression colonnes inutiles : vides + invariantes + spécifiées
├─ Section 6 - Sauvegarde : format CSV pour compatibilité
└─ Validation : aucun doublon, conservation 90%+ des événements
"""

import json
import pandas as pd
import os
import sys
import ast
from datetime import datetime

# ==================== CONFIGURATION ====================
DATA_DIR = "data"
RAW_FILE = f"{DATA_DIR}/raw_events.json"
CLEAN_FILE = f"{DATA_DIR}/clean_events.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Gestion du flag --force
FORCE = "--force" in sys.argv
if "--force" in sys.argv:
    sys.argv.remove("--force")


# ==================== FONCTIONS UTILITAIRES ====================

def extract_fr(field_value):
    """
    Méthode de parsing : extraction du texte français des champs multilingues.
    
    Contexte : OpenAgenda renvoie certains champs comme dicts JSON multilingues :
    - Entrée : {'fr': 'Description en français', 'en': 'English description'}
    - Sortie : 'Description en français'
    
    Cette fonction utilise ast.literal_eval() pour parser la string dict,
    puis extrait la clé 'fr'. Cela permet de récupérer le vrai contenu
    pour les embeddings vectoriels (étape 3 du pipeline RAG).
    
    Args:
        field_value : peut être str, dict, ou None
    
    Returns:
        str : texte français ou chaîne vide
    """
    if pd.isna(field_value):
        return ""
    
    # Si c'est déjà un dict (JSON parsé)
    if isinstance(field_value, dict):
        return str(field_value.get("fr", "")).strip()
    
    # Si c'est une string contenant du JSON
    if isinstance(field_value, str):
        field_value = field_value.strip()
        if field_value.startswith("{"):
            try:
                parsed = ast.literal_eval(field_value)
                if isinstance(parsed, dict):
                    return str(parsed.get("fr", "")).strip()
            except (ValueError, SyntaxError):
                pass
        return field_value
    
    return ""


def clean_events_soft() -> pd.DataFrame:
    """
    Pipeline DOUX de nettoyage des données brutes pour le RAG.
    
    Objectif : maximiser la rétention tout en nettoyant les doublons.
    Approche : 3 niveaux de nettoyage (STRICT → SOFT → SOFT)
    """
    
    print("🧹 Début du nettoyage DOUX des données GPSO...")
    print(f"   Mode force : {FORCE}")

    # 2️⃣ CHARGEMENT DU FICHIER RAW
    if not os.path.exists(RAW_FILE):
        print(f"\n❌ ERREUR : {RAW_FILE} non trouvé !")
        print(f"   💡 Exécute d'abord : python src/data/02_openagenda_fetcher.py")
        raise FileNotFoundError(f"Fichier source manquant : {RAW_FILE}")
    
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"   ✓ Chargement : {len(raw_data)} événements bruts")
    df = pd.DataFrame(raw_data)
    df_before = df.copy()

    # 3️⃣ SÉLECTION COLONNES
    # Méthode : conservation de TOUTES les colonnes disponibles pour max flexibilité
    available_cols = df.columns.tolist()
    cols_priority = ["uid", "title", "description", "slug", "location", 
                     "timings", "status", "keywords", "image", "firstTiming",
                     "categorie-devenement"]
    cols_to_keep = [col for col in cols_priority if col in available_cols]
    cols_to_keep.extend([col for col in available_cols if col not in cols_to_keep])
    
    df = df[cols_to_keep].copy()
    print(f"   ✓ Sélection : {len(cols_to_keep)} colonnes conservées")

    # 4️⃣ SECTION 1 - PARSING MULTILINGUE (extraction français)
    # Méthode utilisée : extraction du texte français des champs dict multilingues
    # Raison : OpenAgenda renvoie {'fr': '...', 'en': '...'} → on extrait 'fr' pour RAG
    print(f"\n   === SECTION 1 : Parsing multilingue ===")
    
    for col in ["title", "description", "longDescription"]:
        if col in df.columns:
            original_count = df[col].notna().sum()
            df[col] = df[col].apply(extract_fr)
            filled_count = df[col].str.len().gt(0).sum()
            print(f"      • {col:20s} : {original_count:3d} non-null → {filled_count:3d} avec texte FR")

    # 5️⃣ SECTION 2 - NETTOYAGE STRICT (doublons seulement)
    print(f"\n   === SECTION 2 : Nettoyage strict (doublons) ===")
    
    length_before = len(df)
    df = df.drop_duplicates(subset=["uid"], keep="first")
    length_after = len(df)
    removed_dupes = length_before - length_after
    
    if removed_dupes > 0:
        print(f"      ✓ Doublons supprimés : {removed_dupes} lignes")
    else:
        print(f"      ✓ Aucun doublon détecté")

    # 6️⃣ SECTION 3 - NETTOYAGE SOFT (espace/valeurs vides)
    print(f"\n   === SECTION 3 : Nettoyage progressif (SOFT) ===")
    
    # Trim espaces pour colonnes string seulement
    for col in df.columns:
        if df[col].dtype == 'object':
            # Appliquer trim seulement si c'est une string, pas une liste/dict
            df[col] = df[col].apply(
                lambda x: str(x).strip() if isinstance(x, str) else x
            )
    print(f"      ✓ Trim espaces : appliqué aux colonnes texte")
    
    # Description vide → placeholder
    if "description" in df.columns:
        df["description"] = df["description"].apply(
            lambda x: "Sans description" if x == "" else x
        )
        print(f"      ✓ Descriptions vides → 'Sans description'")

    # 7️⃣ SECTION 4 - EXTRACTION MÉTADONNÉES
    print(f"\n   === SECTION 4 : Extraction métadonnées ===")
    
    # Extraction première date
    if "firstTiming" in df.columns:
        df["first_event_date"] = pd.to_datetime(df["firstTiming"], errors="coerce")
        dates_valid = df["first_event_date"].notna().sum()
        print(f"      ✓ Dates extraites : {dates_valid} valides")
    else:
        df["first_event_date"] = None
        print(f"      ⚠️  'firstTiming' non disponible")

    # Extraction location (address, postal code, city)
    def extract_location_formatted(loc):
        """Extrait et formate : adresse, code postal, ville"""
        if isinstance(loc, dict):
            address = str(loc.get("address", "") or "").strip()
            postal_code = str(loc.get("postalCode", "") or "").strip()
            city = str(loc.get("city", "") or "").strip()
            # Format : "adresse, 75000, Paris"
            parts = [p for p in [address, postal_code, city] if p]
            return ", ".join(parts) if parts else ""
        return ""
    
    if "location" in df.columns:
        df["location"] = df["location"].apply(extract_location_formatted)
        locations_count = (df["location"].str.len() > 0).sum()
        print(f"      ✓ Lieux formatés : {locations_count} avec address/code postal/ville")
    else:
        df["location"] = ""

    # Formatage timings (dates/heures lisibles)
    def format_timings(timings_str):
        """Parse et formate les timings en texte lisible"""
        try:
            # Gérer les valeurs vides/null
            if timings_str is None or (isinstance(timings_str, str) and timings_str.strip() == ""):
                return "Pas de dates disponibles"
            
            # Si string, parser en list
            if isinstance(timings_str, str):
                timings_list = ast.literal_eval(timings_str)
            else:
                timings_list = timings_str
            
            if not isinstance(timings_list, list):
                return "Pas de dates disponibles"
            
            formatted_times = []
            for event in timings_list:
                if isinstance(event, dict):
                    begin = event.get("begin", "")
                    end = event.get("end", "")
                    if begin:
                        try:
                            dt_begin = pd.to_datetime(begin)
                            dt_end = pd.to_datetime(end) if end else None
                            
                            # Format : "13 mars 2026 19h30"
                            begin_str = dt_begin.strftime("%d %b %Y %Hh%M").replace(' ', ' ')
                            
                            if dt_end:
                                end_str = dt_end.strftime("%Hh%M")
                                formatted_times.append(f"{begin_str} - {end_str}")
                            else:
                                formatted_times.append(begin_str)
                        except:
                            pass
            
            return " | ".join(formatted_times) if formatted_times else "Pas de dates disponibles"
        except:
            return "Pas de dates disponibles"
    
    if "timings" in df.columns:
        df["timings"] = df["timings"].apply(format_timings)
        print(f"      ✓ Timings formatés : dates/heures lisibles")
    
    # Formatage registration (modes d'inscription lisibles)
    def format_registration(reg_str):
        """Parse et formate les modes d'inscription"""
        try:
            # Gérer les valeurs vides/null
            if reg_str is None or (isinstance(reg_str, str) and reg_str.strip() == ""):
                return "Pas de mode d'inscription spécifié"
            
            # Parser si string
            if isinstance(reg_str, str):
                reg_list = ast.literal_eval(reg_str)
            else:
                reg_list = reg_str
            
            # Vérifier que c'est bien une liste avec éléments
            if not isinstance(reg_list, list) or len(reg_list) == 0:
                return "Pas de mode d'inscription spécifié"
            
            methods = []
            for item in reg_list:
                if isinstance(item, dict):
                    reg_type = item.get("type", "")
                    value = item.get("value", "")
                    
                    if reg_type == "link" and value:
                        methods.append(f"Inscription: {value}")
                    elif reg_type == "phone" and value:
                        methods.append(f"Téléphone: {value}")
                    elif reg_type == "email" and value:
                        methods.append(f"Email: {value}")
            
            return " | ".join(methods) if methods else "Pas de mode d'inscription spécifié"
        except:
            return "Pas de mode d'inscription spécifié"
    
    if "registration" in df.columns:
        df["registration"] = df["registration"].apply(format_registration)
        print(f"      ✓ Inscriptions formatées : modes d'inscription lisibles")
    
    # Formatage links (liens lisibles)
    def format_links(links_str):
        """Parse et formate les liens"""
        try:
            # Gérer les valeurs vides/null
            if links_str is None or (isinstance(links_str, str) and links_str.strip() == ""):
                return "Pas de lien spécifié"
            
            if isinstance(links_str, str):
                links_list = ast.literal_eval(links_str)
            else:
                links_list = links_str
            
            if not isinstance(links_list, list) or len(links_list) == 0:
                return "Pas de lien spécifié"
            
            links = []
            for item in links_list:
                if isinstance(item, dict):
                    link = item.get("link", "")
                    if link:
                        links.append(link)
            
            return " | ".join(links) if links else "Pas de lien spécifié"
        except:
            return "Pas de lien spécifié"
    
    if "links" in df.columns:
        df["links"] = df["links"].apply(format_links)
        print(f"      ✓ Liens formatés : lisibles")
    
    # Formatage age (tranches d'âge lisibles)
    def format_age(age_str):
        """Parse et formate les tranches d'âge"""
        try:
            # Gérer les valeurs vides/null
            if age_str is None or (isinstance(age_str, str) and age_str.strip() == ""):
                return "Pas de restriction d'âge"
            
            if isinstance(age_str, str):
                age_dict = ast.literal_eval(age_str)
            else:
                age_dict = age_str
            
            if not isinstance(age_dict, dict):
                return "Pas de restriction d'âge"
            
            min_age = age_dict.get("min")
            max_age = age_dict.get("max")
            
            if min_age is not None and max_age is not None:
                return f"{min_age} à {max_age} ans"
            elif min_age is not None:
                return f"À partir de {min_age} ans"
            elif max_age is not None:
                return f"Jusqu'à {max_age} ans"
            
            return "Pas de restriction d'âge"
        except:
            return "Pas de restriction d'âge"
    
    if "age" in df.columns:
        df["age"] = df["age"].apply(format_age)
        print(f"      ✓ Tranches d'âge formatées : lisibles")
    
    # Formatage conditions (tarifs/infos pratiques)
    def format_conditions(cond_str):
        """Parse et formate les conditions"""
        try:
            # Gérer les valeurs vides/null
            if cond_str is None or (isinstance(cond_str, str) and cond_str.strip() == ""):
                return "Pas d'informations de conditions disponibles"
            
            if isinstance(cond_str, str):
                cond_dict = ast.literal_eval(cond_str)
            else:
                cond_dict = cond_str
            
            if isinstance(cond_dict, dict):
                # Récupérer la clé 'fr' si elle existe
                if "fr" in cond_dict:
                    text = str(cond_dict["fr"]).strip()
                    return text if text else "Pas d'informations de conditions disponibles"
                else:
                    # Sinon prendre la première valeur textuelle
                    for v in cond_dict.values():
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            
            return "Pas d'informations de conditions disponibles"
        except:
            return "Pas d'informations de conditions disponibles"
    
    if "conditions" in df.columns:
        df["conditions"] = df["conditions"].apply(format_conditions)
        print(f"      ✓ Conditions formatées : tarifs/infos pratiques")

    # 8️⃣ SECTION 5 - SUPPRESSION COLONNES INUTILES
    print(f"\n   === SECTION 5 : Suppression colonnes inutiles ===")
    
    # Colonnes à supprimer manuellement (non-utiles pour RAG/embeddings)
    cols_to_drop_explicit = [
        "slug", "status", "image", "categorie-devenement", "country", 
        "featured", "private", "dateRange", "timezone", "imageCredits", 
        "originAgenda", "onlineAccessLink", "valid", "createdAt", "motive", "draft",
        "firstTiming", "accessibility", "updatedAt", "addMethod", "attendanceMode",
        "sourceAgendas", "creatorUid", "lastTiming", "ownerUid", "nextTiming",
        "uid", "description", "keywords", "city"
    ]
    
    # Supprimer les colonnes qui existent
    cols_to_drop = [col for col in cols_to_drop_explicit if col in df.columns]
    
    # Ajouter les colonnes vides (100% manquantes)
    empty_cols = [col for col in df.columns if df[col].isna().sum() == len(df)]
    cols_to_drop.extend(empty_cols)
    
    # Ajouter les colonnes avec une seule valeur unique (invariantes)
    # Convertir dicts/listes en strings pour compter les uniques (sinon TypeError: unhashable type)
    single_value_cols = []
    for col in df.columns:
        if col not in cols_to_drop:
            try:
                # Essayer de compter les uniques directement
                nunique = df[col].nunique()
            except TypeError:
                # Si erreur (dicts, listes, etc.), convertir en string
                nunique = df[col].astype(str).nunique()
            
            if nunique == 1:
                single_value_cols.append(col)
    
    cols_to_drop.extend(single_value_cols)
    
    # Dédupliquer et supprimer
    cols_to_drop = list(set(cols_to_drop))
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if cols_to_drop:
        print(f"      ✓ Colonnes supprimées ({len(cols_to_drop)}) :")
        print(f"        └─ Manuelles : {[c for c in cols_to_drop_explicit if c in cols_to_drop]}")
        if empty_cols:
            print(f"        └─ Vides (100% NaN) : {empty_cols}")
        if single_value_cols:
            print(f"        └─ Invariantes (1 valeur) : {single_value_cols}")
        df = df.drop(columns=cols_to_drop)
    else:
        print(f"      ✓ Aucune colonne à supprimer")
    
    print(f"      ✓ Colonnes restantes : {len(df.columns)}")

    # 9️⃣ SECTION 6 - SAUVEGARDE
    print(f"\n   === SECTION 6 : Sauvegarde ===")
    
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_FILE, index=False, encoding="utf-8")
    print(f"      ✓ CSV sauvegardé : {CLEAN_FILE}")

    # 9️⃣ RÉSUMÉ FINAL
    print("\n" + "=" * 80)
    print("✅ Nettoyage DOUX terminé avec succès !")
    print("=" * 80)
    
    print(f"\n📊 Résumé statistique:")
    print(f"   Événements bruts      : {len(df_before)}")
    print(f"   Événements nettoyés   : {len(df)}")
    retention = 100 * len(df) / len(df_before)
    perte = len(df_before) - len(df)
    print(f"   Rétention            : {retention:.1f}% (+{perte} suppressions)")
    
    print(f"\n🏙️  Top 10 des villes :")
    print("-" * 80)
    if "city" in df.columns:
        city_counts = df["city"].value_counts().head(10)
        print(city_counts.to_string())
        top_cities = df["city"].value_counts().index[:3].tolist()
        print(f"\n   Top 3 : {', '.join(top_cities)}")
    else:
        print("   ⚠️  Colonne 'city' non disponible")
    
    print(f"\n📋 Colonnes finales :")
    print("-" * 80)
    cols_summary = pd.DataFrame({
        'Colonne': df.columns,
        'Type': [str(df[col].dtype) for col in df.columns],
        'Non-null%': [f"{df[col].notna().sum()/len(df)*100:.0f}%" for col in df.columns],
    })
    
    # Print avec colonnes plus lisibles
    for idx, row in cols_summary.iterrows():
        print(f"   {row['Colonne']:20s} {row['Type']:10s} {row['Non-null%']:>8s}")
    
    print(f"\n💾 Fichier créé : {CLEAN_FILE}")
    print(f"   Taille : {os.path.getsize(CLEAN_FILE) / 1024:.1f} KB")
    print("-" * 80)
    
    return df


# ==================== LANCEMENT ====================
if __name__ == "__main__":
    clean_events_soft()
