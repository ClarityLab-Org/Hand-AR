"""Sketchfab API client for searching and downloading 3D models."""

import os
import time
from typing import Dict, List, Optional

import requests
from requests import RequestException

from handarm.config import SKETCHFAB_TOKEN
from handarm.scanning.database import load_index, save_index
from handarm.scanning.local_scanner import scan_local

# Graceful handling of missing token (Bug fix: was raising ValueError)
_token_available: bool = bool(SKETCHFAB_TOKEN)
_REQUEST_TIMEOUT = 30
if not _token_available:
    print(" No SKETCHFAB_TOKEN found in .env. Web search/download disabled.")
    print("  Local model loading still works normally.")


def search_web(query: str = "car") -> List[Dict]:
    """
    Searches the Sketchfab API v3 for downloadable 3D models matching a query.

    Args:
        query (str): Search term for Sketchfab models. Defaults to "car".

    Returns:
        list: List of downloadable web model records containing name, UID, and viewer URL.
    """
    if not _token_available:
        return []
        
    results = []
    url = "https://api.sketchfab.com/v3/search"
    params = {
        "q": query,
        "type": "models"
    }

    try:
        res = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
        res.raise_for_status()
        data = res.json()

        for model in data.get("results", []):            
            if not model.get("isDownloadable", False):
                continue
            uid = model.get("uid")
            name = model.get("name")
            viewer_link = f"https://sketchfab.com/3d-models/{uid}"
            results.append({
                "name": name,
                "uid": uid,
                "viewer": viewer_link,
                "source": "web"
            })
    except (RequestException, ValueError) as e:
        print("Web search failed:", e)
    return results


def download_glb(model: Dict, save_folder: str) -> Optional[str]:
    """
    Downloads a GLB model asset from Sketchfab using the API token.

    Args:
        model (dict): Model dictionary containing source, uid, and name metadata.
        save_folder (str): Directory path to save the downloaded model.

    Returns:
        str: Absolute file path to the saved local GLB file, or None if download fails.
    """
    if model.get("source") != "web":
        return model.get("path")
        
    if not _token_available:
        return None

    try:
        uid = model.get("uid")
        if not uid:
            print("No UID found")
            return None

        headers = {
            "Authorization": f"Token {SKETCHFAB_TOKEN}"
        }

        url = f"https://api.sketchfab.com/v3/models/{uid}/download"
        res = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        res.raise_for_status()
        data = res.json()

        if "glb" in data:
            download_url = data["glb"]["url"]
        elif "zip" in data:
            download_url = data["zip"]["url"]
        else:
            print("No downloadable format..")
            return None

        name = model["name"].replace(" ", "_")
        save_path = os.path.abspath(os.path.join(save_folder, name + ".glb"))

        print(f"Downloading {name}...")

        r = requests.get(download_url, stream=True, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        print(f"Saved to {save_path}")

        database = load_index()
        updated = False
        for item in database.get("results", []):
            if item.get("uid") == uid:
                item["source"] = "local"
                item["path"]   = save_path
                updated = True
                break
        if updated:
            save_index(database)
            print(f"Database updated: {name} is now local")

        return save_path

    except (OSError, RequestException, ValueError) as e:
        print("Download failed:", e)
        return None


def unified_search(folder: str, query: str = "car", use_cache: bool = True) -> List[Dict]:
    """
    Combines local model files and online Sketchfab models into a unified search result.

    Args:
        folder (str): Local models directory path.
        query (str): Search query keyword. Defaults to "car".
        use_cache (bool): If True, returns cached results when queries match database.json.

    Returns:
        list: Deduplicated list of local and web search result records.
    """
    database = load_index()

    if use_cache and database:
        if database.get("query") == query:
            return database["results"]

    local_results = scan_local(folder)
    web_results = search_web(query)

    combined = local_results + web_results

    unique = {}
    for item in combined:
        key = item.get("path") or item.get("uid")
        unique[key] = item

    combined = list(unique.values())

    data = {
        "query": query,
        "timestamp": time.time(),
        "results": combined
    }

    save_index(data)
    return combined
