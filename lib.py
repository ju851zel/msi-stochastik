import requests
import json
import matplotlib.pyplot as plt
import plotly.express as px
import re
import numpy as np


def pretty_print(pretty_print_content):
    print(json.dumps(pretty_print_content, indent=4, sort_keys=True))


BASE_URL = 'https://swapi.dev/api/'
AUTH_KEY = ''
headers = {'Authorization': 'Bearer ' + AUTH_KEY}


#
def sortfunc(elem):
    return int(elem["passengers"].replace(',', ''))


def basic_data():
    # Water and amount of population korreliert?
    page = 1
    response = requests.get(BASE_URL + "planets").content
    planets = []
    while requests.get(BASE_URL + f"planets/?page={page}").ok:
        content = requests.get(BASE_URL + f"planets/?page={page}").content
        planets += json.loads(content)["results"]
        page += 1

    planet_population = []
    planet_diameter = []
    planets
    planet_gravity = []
    planet_names = []
    planet_orbital = []

    planets = list(filter(lambda pl: pl["population"] != "unknown" and
                                     pl["gravity"] != "unknown" and
                                     pl["gravity"] != "standard",
                          planets))
    planets.sort(key=sortfunc)

    for planet in planets:
        planet_names.append(planet["name"])
        planet_diameter.append(planet["diameter"])
        planet_gravity.append(float(re.sub("[^\.1-9]", "", planet["gravity"])))
        # planet_gravity.append(planet["gravity"])
        planet_population.append(planet["population"])
    amount_planets = len(planets)
    # print(f"Anzahl der Planeten : {amount_planets}")
    # print(f"diameter: {planet_diameter}")
    print(f"gravity: {planet_gravity}")

    plt.scatter(planet_gravity, planet_population)
    plt.ylabel('Population')
    plt.xlabel('Gravity')
    plt.grid(True)

    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()


def test():
    # welche imperialen raumschiffe
    # welche schiffe transportieren am meisten passagiere mit kleiner crew am günstigsten
    page = 1
    ships = []
    while requests.get(BASE_URL + f"starships/?page={page}").ok:
        content = requests.get(BASE_URL + f"starships/?page={page}").content
        ships += json.loads(content)["results"]
        page += 1

    newShips = []
    for ship in ships:
        if ship["passengers"] == "n/a" or ship["passengers"] == "unknown" or ship["cost_in_credits"] == "unknown":
            pass
        else:
            newShips.append(ship)
    ships = newShips
    ships.sort(key=sortfunc)
    passengers = []
    cost_in_credits = []
    for ship in ships:
        # passengers.append(int(re.sub("[^0-9]", "", ship["passengers"])))
        passengers.append(int(ship["passengers"].replace(',', '')))
        cost_in_credits.append(int(ship["cost_in_credits"]))

    passengers.pop()
    cost_in_credits.pop()
    # passengers = list(map(lambda pl: pl["passengers"], ships))
    # cost_in_credits = list(map(lambda pl: pl["cost_in_credits"], ships))

    pretty_print(passengers)
    pretty_print(cost_in_credits)

    plt.scatter(passengers, cost_in_credits)
    plt.ylabel('cost_in_credits')
    plt.xlabel('passengers')
    plt.grid(True)
    plt.ticklabel_format(useOffset=False, style='plain')

    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    plt.show()

    # basic_data()


def testWahlen():
    ships = []
    res = requests.get(f"https://api.dawum.de").content
    res = json.loads(res)
    parliaments = res["Parliaments"]
    institutes = res["Institutes"]
    parties = res["Parties"]
    surveys = res["Surveys"]

    bundestagsumfragen = {}
    for survey in surveys.values():
        if survey["Parliament_ID"] == "0":
            if survey["Institute_ID"] not in bundestagsumfragen:
                bundestagsumfragen[survey["Institute_ID"]] = []
            bundestagsumfragen[survey["Institute_ID"]].append(survey)
    # for kay, value in bundestagsumfragen.items():
    #     pretty_print(f"{kay, len(value) }")

    surveys = bundestagsumfragen["2"]
    # pretty_print(surveys)

    data = {}
    for survey in surveys:
        data[survey["Date"]] = {
            "cdu": survey["Results"]["1"],
            "spd": survey["Results"]["2"],
            "inc": ""
        }

    # pretty_print(dates)
    # pretty_print(cduPoints)
    # plt.scatter(dates, cduPoints)
    # plt.ylabel('cdu werte')
    # plt.xlabel('datum')
    # plt.grid(True)
    # plt.ticklabel_format(useOffset=False, style='plain')
    #
    # plt.autoscale()
    # plt.yticks(planet_population,
    #            planet_names)
    # plt.show()

    res = requests.get(f"https://api.corona-zahlen.org/germany/history/incidence").content
    res = json.loads(res)["data"]
    for dataPoint in res:
        if dataPoint["date"][0:10] in data:
            data[dataPoint["date"][0:10]]["inc"] = dataPoint[
                "weekIncidence"]  # str("%.2f" % dataPoint["deaths"]).replace(".", ",")

    resDeath = []
    resCDU = []
    resSPD = []
    resDate = []
    result = []
    for key, value in data.items():
        if value["inc"] != "":
            resDate.append(key)
            resCDU.append((value["cdu"]))
            resDeath.append((value["inc"]))
            resSPD.append((value["spd"]))
            result.append(value)

    resCDU.reverse()
    resDate.reverse()
    resDeath.reverse()
    resSPD.reverse()
    pretty_print(resDate)
    # Scatterplot
    plt.scatter(
        x=resDate,
        y=resCDU,
        c="blue")

    plt.title("CDU Umfragewerte")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])
    plt.show()

    plt.scatter(
        x=resDate,
        y=resDeath,
        c="red")
    plt.title("Corona Inzidenzen")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])

    plt.show()

    plt.scatter(
        x=resDate,
        y=resSPD,
        c="green")
    plt.title("SPD Umfragewerte")
    plt.xticks(rotation=90)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60])

    plt.show()


testWahlen()
