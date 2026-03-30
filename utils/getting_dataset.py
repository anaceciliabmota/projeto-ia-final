import requests, os
from PIL import Image
from io import BytesIO
import re

HEADERS = {"User-Agent": "MeuProjetoIA/1.0 (anaceciliabezerra120@gmail.com)"}
PREFERRED_COUNTRIES = ["GB", "US", "XW"]
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRACKS_PATH = os.path.join(DATA_DIR, "tracks.json")
COVERS_DIR = os.path.join(DATA_DIR, "covers")

def normalize_title(text):
    return (
        text.strip()
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
        .lower()
    )

def normalize_track_title(text):
    cleaned = normalize_title(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

def score_release_group(group):
    primary_type = group.get("primary-type")
    secondary = set(group.get("secondary-types") or [])
    first_release_date = group.get("first-release-date") or "9999-99-99"

    primary_score = 0 if primary_type == "Album" else 1
    secondary_score = 0 if not secondary else 1
    unwanted_secondary = 1 if {"Compilation", "Live", "Remix"} & secondary else 0
    return (primary_score, unwanted_secondary, secondary_score, first_release_date)

def get_beatles_releases():
    releases = {}
    offset = 0
    while True:
        resp = requests.get(
            "https://musicbrainz.org/ws/2/release-group",
            params={
                "artist": "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d",
                "type": "album",
                "fmt": "json",
                "limit": 100,
                "offset": offset,
            },
            headers=HEADERS,
        )
        data = resp.json()
        groups = data.get("release-groups", [])
        if not groups:
            break
        for group in groups:
            title = group["title"]
            normalized = normalize_title(title)
            candidate = {"id": group["id"], "score": score_release_group(group)}
            current = releases.get(normalized)
            if current is None or candidate["score"] < current["score"]:
                releases[normalized] = candidate
        offset += len(groups)
        if offset >= data.get("release-group-count", 0):
            break
    return {title: payload["id"] for title, payload in releases.items()}

def pick_best_release(releases):
    def score(release):
        status = release.get("status")
        status_score = 0 if status == "Official" else 1

        country = release.get("country")
        if country in PREFERRED_COUNTRIES:
            country_score = PREFERRED_COUNTRIES.index(country)
        else:
            country_score = len(PREFERRED_COUNTRIES)

        # Datas ausentes vao para o fim.
        date_score = release.get("date") or "9999-99-99"
        return (status_score, country_score, date_score)

    if not releases:
        raise ValueError("Release group sem releases disponiveis")
    return sorted(releases, key=score)[0]

def get_tracklist(release_group_id):
    # 1. escolhe uma release oficial e preferencialmente UK/US.
    rg = requests.get(
        f"https://musicbrainz.org/ws/2/release-group/{release_group_id}",
        params={"inc": "releases", "fmt": "json"},
        headers=HEADERS,
    ).json()
    best_release = pick_best_release(rg["releases"])
    release_id = best_release["id"]

    # 2. pega as faixas dessa release
    release = requests.get(
        f"https://musicbrainz.org/ws/2/release/{release_id}",
        params={"inc": "recordings", "fmt": "json"},
        headers=HEADERS,
    ).json()

    tracks = []
    seen_in_album = set()
    for medium in release["media"]:
        for track in medium["tracks"]:
            duration = track.get("length")
            if duration is None:
                duration = track.get("recording", {}).get("length")
            title = track["title"]
            norm_title = normalize_track_title(title)
            if norm_title in seen_in_album:
                continue
            seen_in_album.add(norm_title)
            tracks.append({
                "position": track["position"],
                "title": title,
                "duration_ms": duration,
            })
    return tracks

def dedupe_tracks_across_albums(dataset):
    seen_global = set()
    deduped = {}
    for album in STUDIO_ALBUMS:
        album_data = dataset.get(album)
        if not album_data:
            continue
        filtered_tracks = []
        for track in album_data.get("tracks", []):
            norm_title = normalize_track_title(track.get("title", ""))
            if not norm_title or norm_title in seen_global:
                continue
            seen_global.add(norm_title)
            filtered_tracks.append(track)
        deduped[album] = {
            "cover": album_data.get("cover"),
            "tracks": filtered_tracks,
        }
    return deduped

def download_cover(release_group_id, title, output_dir=COVERS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://coverartarchive.org/release-group/{release_group_id}/front"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"[sem capa] {title}")
        return None
    img = Image.open(BytesIO(resp.content))
    path = os.path.join(output_dir, f"{title.replace('/', '-')}.jpg")
    img.save(path)
    print(f"[ok] {title} -> {path}")
    return path

STUDIO_ALBUMS = [
    "Please Please Me", "With The Beatles", "A Hard Day's Night",
    "Beatles for Sale", "Help!", "Rubber Soul", "Revolver",
    "Sgt. Pepper's Lonely Hearts Club Band", "Magical Mystery Tour",
    "The Beatles", "Yellow Submarine", "Let It Be", "Abbey Road",
]

if __name__ == "__main__":
    import time, json

    os.makedirs(DATA_DIR, exist_ok=True)
    dataset_path = TRACKS_PATH
    dataset = json.load(open(dataset_path, encoding="utf-8")) if os.path.exists(dataset_path) else {}

    all_releases = get_beatles_releases()
    releases = {}
    for title in STUDIO_ALBUMS:
        normalized = normalize_title(title)
        if normalized in all_releases:
            releases[title] = all_releases[normalized]

    for title, release_id in releases.items():
        print(f"\n{title}")
        tracklist = get_tracklist(release_id)
        cover_path = download_cover(release_id, title)
        # Mantem capa antiga se o download falhar.
        current_cover = dataset.get(title, {}).get("cover")
        dataset[title] = {"cover": cover_path or current_cover, "tracks": tracklist}

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # Remove entradas fora da lista oficial e deduplica musicas entre albuns.
    dataset = {title: dataset[title] for title in STUDIO_ALBUMS if title in dataset}
    dataset = dedupe_tracks_across_albums(dataset)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[ok] tracks.json com {len(dataset)} álbuns")