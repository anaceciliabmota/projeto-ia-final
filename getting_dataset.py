import requests, os
from PIL import Image
from io import BytesIO

HEADERS = {"User-Agent": "MeuProjetoIA/1.0 (anaceciliabezerra120@gmail.com)"}

def get_beatles_releases():
    resp = requests.get(
        "https://musicbrainz.org/ws/2/release-group",
        params={"artist": "b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d", "type": "album", "fmt": "json", "limit": 100},
        headers=HEADERS,
    )
    return {g["title"]: g["id"] for g in resp.json()["release-groups"]}

def get_tracklist(release_group_id):
    # 1. pega a primeira release do grupo
    rg = requests.get(
        f"https://musicbrainz.org/ws/2/release-group/{release_group_id}",
        params={"inc": "releases", "fmt": "json"},
        headers=HEADERS,
    ).json()
    release_id = rg["releases"][0]["id"]

    # 2. pega as faixas dessa release
    release = requests.get(
        f"https://musicbrainz.org/ws/2/release/{release_id}",
        params={"inc": "recordings", "fmt": "json"},
        headers=HEADERS,
    ).json()

    tracks = []
    for medium in release["media"]:
        for track in medium["tracks"]:
            tracks.append({
                "position": track["position"],
                "title": track["title"],
                "duration_ms": track["length"],
            })
    return tracks

def download_cover(release_group_id, title, output_dir="covers"):
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

    dataset_path = "dataset.json"
    dataset = json.load(open(dataset_path, encoding="utf-8")) if os.path.exists(dataset_path) else {}

    all_releases = get_beatles_releases()
    releases = {t: id for t, id in all_releases.items() if t in STUDIO_ALBUMS}

    for title, release_id in releases.items():
        if title in dataset:
            print(f"[skip] {title}")
            continue

        print(f"\n{title}")
        tracklist = get_tracklist(release_id)
        cover_path = download_cover(release_id, title)
        dataset[title] = {"cover": cover_path, "tracks": tracklist}

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"\n[ok] dataset.json com {len(dataset)} álbuns")