import requests

OMDB_API_KEY = "4e3e9ffa"
TMDB_API_KEY = "e7a874655df2bb6d279806d619e16875"  # obtenha em themoviedb.org/settings/api

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_HORROR_GENRE_ID = 27  # ID fixo do gênero Terror no TMDB


def buscar_filmes_terror(quantidade=200, ordenar_por="popularity.desc", min_votos=200):
    """
    Retorna lista de filmes de terror via TMDB Discover.

    ordenar_por: 'popularity.desc', 'vote_average.desc', 'vote_count.desc', 'release_date.desc'
    min_votos: filtra filmes com poucos votos (ruído no dataset)
    """
    filmes = []
    page = 1

    while len(filmes) < quantidade:
        data = requests.get(
            f"{TMDB_BASE}/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "with_genres": TMDB_HORROR_GENRE_ID,
                "language": "pt-BR",
                "sort_by": ordenar_por,
                "vote_count.gte": min_votos,
                "page": page,
            },
            timeout=10,
        ).json()

        resultados = data.get("results", [])
        if not resultados:
            break

        for filme in resultados:
            filmes.append({
                "titulo": filme.get("title", "N/A"),
                "titulo_original": filme.get("original_title", "N/A"),
                "ano": (filme.get("release_date") or "")[:4] or "N/A",
                "sinopse": filme.get("overview") or "N/A",
                "nota_tmdb": filme.get("vote_average", "N/A"),
                "votos": filme.get("vote_count", "N/A"),
                "poster": f"https://image.tmdb.org/t/p/w500{filme['poster_path']}" if filme.get("poster_path") else "",
                "tmdb_id": filme.get("id"),
            })
            if len(filmes) >= quantidade:
                break

        page += 1

    return filmes


if __name__ == "__main__":
    filmes = buscar_filmes_terror()
    # for filme in filmes:
    #     print(filme["titulo"])
    #     print(filme["titulo_original"])
    #     print(filme["ano"])
    #     print(filme["nota_tmdb"])
    #     print(filme["poster"])
    #     print("--------------------------------")
    
    print("por media de votos")
    filmes = buscar_filmes_terror(ordenar_por="vote_average.desc", min_votos=200)
    for filme in filmes:
        print(filme["titulo"])
        print(filme["titulo_original"])
        print(filme["ano"])
        print(filme["nota_tmdb"])
        print(filme["poster"])
        print(filme["sinopse"])
        print("--------------------------------")
