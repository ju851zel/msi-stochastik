import requests
import json
import matplotlib.pyplot as plt


def pretty_print(pretty_print_content):
    print(json.dumps(pretty_print_content, indent=4, sort_keys=True))


BASE_URL = 'https://the-one-api.dev/v2/'
AUTH_KEY = 'mxj-FoJ3Zv89Z4068cAO'
headers = {'Authorization': 'Bearer ' + AUTH_KEY}


def basic_data():
    books = requests.get(BASE_URL + "book", headers=headers).content
    books = json.loads(books)["docs"]
    amount_of_book = len(books)
    print(f"Anzahl der LOTR Bücher : {amount_of_book}")

    movies = requests.get(BASE_URL + "movie", headers=headers).content
    movies = json.loads(movies)["docs"]
    amount_of_movies = len(movies)
    print(f"Anzahl der Filme : {amount_of_movies - 2}")

    chars = requests.get(BASE_URL + "character", headers=headers).content
    chars = json.loads(chars)["docs"]
    amount_of_chars = len(chars)
    print(f"Anzahl der Charactere : {amount_of_chars}")

    quotes = requests.get(BASE_URL + "quote?limit=3000", headers=headers).content
    quotes = json.loads(quotes)["docs"]
    amount_of_quotes = len(quotes)
    print(f"Anzahl der film zitate : {amount_of_quotes}")

    chapters = requests.get(BASE_URL + "chapter?limit=3000", headers=headers).content
    chapters = json.loads(chapters)["docs"]
    amount_of_chapters = len(chapters)
    print(f"Anzahl der buch kapitel : {amount_of_chapters}")

    # pretty_print(chapters)

    # oscars = []
    # for movie in movies["docs"]:
    #     oscars.append({
    #         "name": movie["name"],
    #         "oscars_nom": movie["academyAwardNominations"],
    #         "oscars_win": movie["academyAwardWins"]
    #     })
    #
    # oscars_noms = []
    # for movie in oscars:
    #     oscars_noms.append(movie["oscars_nom"])
    #
    # plt.plot(oscars_noms)
    # plt.ylabel('Oscars Nominiert')
    # plt.xlabel('Film')
    # plt.autoscale()
    # plt.xticks([1, 2, 3, 4, 5, 6],
    #            [oscars[0]["name"],
    #             oscars[1]["name"],
    #             oscars[2]["name"],
    #             oscars[3]["name"],
    #             oscars[4]["name"],
    #             oscars[5]["name"],
    #             ])
    # plt.show()
    # print(f"Oscars : {oscars}")


basic_data()
