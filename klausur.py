# Kernel Density Estimation, KDE
import json
import math
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
import statsmodels.stats.weightstats
import matplotlib.pyplot as plt
import numpy as np
import scipy
# boxplot
import seaborn as seaborn
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import hypergeom
from scipy.stats import poisson


def pretty_print(pretty_print_content):
    print(json.dumps(pretty_print_content, indent=2, sort_keys=True))


def regression(a, b):
    r = empirischeKorrellationsKoeffizient(a, b)
    sy = empirischeNminus1Standardabweichung(b)
    sx = empirischeNminus1Standardabweichung(a)
    xm = arithmMittel(a)
    ym = arithmMittel(b)
    k = r * (sy / sx)
    d = ym - (k * xm)
    print("f(x) = k * x + d")
    print("k = rx,y * (sy/sx)", k)
    print("d = ym - k * xm = ", d)
    plt.scatter(a, b)
    x = np.linspace(min(a), max(a), 100)
    y = k * x + d
    plt.plot(x, y, '-r', label='y=2x+1')
    plt.ylabel('x first')
    plt.xlabel('y second')
    plt.grid(True)
    plt.show()


def absoluteHaufigkeit(array):
    res = {}
    absolut = {}
    for item in array:
        if item not in res.keys():
            res[item] = 0
        res[item] = res[item] + 1
    for key, val in res.items():
        absolut[key] = val
    plt.bar(absolut.keys(), absolut.values())
    plt.ylabel('absolut Haufigkeit')
    plt.xlabel('nums')
    plt.grid(True)
    plt.show()
    print("Absolute Haufigkeiten: ")
    pretty_print(absolut)
    return absolut


def relativeHaufigkeit(array):
    res = {}
    relative = {}
    for item in array:
        if item not in res.keys():
            res[item] = 0
        res[item] = res[item] + 1
    for key, val in res.items():
        relative[key] = val / len(array)
    plt.bar(relative.keys(), relative.values())
    plt.ylabel('relative Haufigkeit')
    plt.xlabel('nums')
    plt.grid(True)
    plt.show()
    print("Relative Haufigkeiten: ")
    pretty_print(relative)
    return relative


def empirscheVerteilungsfunktionDraw(array):
    y = []
    array = np.array(array)
    for i in array:
        y += [len(array[array <= i]) / len(array)]
    y = np.array(y)
    xs = list(set(array))
    ys = list(set(y))
    ys, xs = zip(*sorted(zip(ys, xs)))
    plt.bar(xs, ys)
    plt.ylabel('Wahrscheinlichkeit')
    plt.xlabel('nums')
    plt.grid(True)
    plt.show()
    res = {}
    j = 0
    for i in xs:
        res[i] = ys[j]
        j += 1
    return res


def mittelwert(array):
    x = np.mean(array)
    print("Mittelwert: ", x)
    return x


def arithmMittel(array):
    x = np.mean(array)
    print("ArithMittel: ", x)
    return x


def median(array):
    x = np.median(array)
    print("Median: ", x)
    return x


def modal(array):
    # haufigste auftretende Wert
    x = scipy.stats.mode(array).mode
    print("Modal (haufigst auftret Wert): ", x)
    return x


def pQuantil(array, percentile):
    res = 0
    array = sorted(array)
    index = int(len(array) * (percentile / 100))
    if percentile % 2 == 0:
        res = 0.5 * (array[index - 1] + array[index])
    else:
        index = math.ceil(index)
        res = array[index]
    # x = np.percentile(array, percentile)
    print(percentile, "-Quartil: ", res)
    return res


def empirischNminus1Varianz(array):
    # n-1
    x = np.var(array, ddof=1)
    print("empirische Varianz: ", x)
    return x


def empirischeNminus1Standardabweichung(array):
    # n-1
    x = np.std(array, ddof=1)
    print("empirische Standardabweichung: ", x)
    return x


def diskreteVarianz(array, weights):
    # n
    x = statsmodels.stats.weightstats.DescrStatsW(array, weights=weights, ddof=0).var
    print("Varianz einer Zufallsgrosse: ", x)
    return x


def diskreteErwartungswert(array, weights):
    # n
    x = np.average(array, weights=weights)
    print("Erwartungswert einer Zufallsgrosse: ", x)
    return x


def diskreteStandardabweichung(array, weights):
    # n
    x = statsmodels.stats.weightstats.DescrStatsW(array, weights=weights, ddof=0).std
    print("normale Standardabweichung: ", x)
    return x


def spannweite(array):
    x = max(array) - min(array)
    print("Spannweite: ", x)
    return x


def interquartilabstandIRQ(array):
    x = scipy.stats.iqr(array)
    print("Interquartilabstand: ", x)
    return x


def empirischeSchiefe(array):  # skewness
    x = scipy.stats.skew(array)
    print("empirische Schiefe: ", x)
    return x


def empirischeWoelbung(array):  # kurtosis
    x = scipy.stats.kurtosis(array)
    print("empirische Woelbung: ", x)
    return x


def empirischeKorrellationsKoeffizient(a, b):
    print("empirischer korrelationskoeff: ", np.corrcoef(a, b)[0][1])
    print("pearson korrelationskoeff (p-wert): ", pearsonr(a, b))
    return np.corrcoef(a, b)[0][1]


def empirischeKovarianz(a, b):
    data = np.array([a, b])
    print("empirischer kovarianz ", np.cov(data, bias=True))


# CFD cumulative distribution function
def empirischeVerteilungsfunktion(a, b):
    data = np.array([a, b])
    print("empirischer Verteilungsfunktion ", np.cumsum(data))


# CFD cumulative distribution function
def kde(a, b):
    pass  # todo


def binomial_verteilung(n, p, array, verteilung=False):
    mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
    r_values = list(range(n + 1))
    dist = [binom.pmf(r, n, p) for r in r_values]
    if verteilung:
        for i in range(n + 1):
            print(str(r_values[i]) + "\t" + str(dist[i]))

    sum = 0
    for i in array:
        sum += dist[i]
    print("Wahrsch: ", array, " mal: ", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)


def hyper_geometrisch_verteilung(array, nStichprobe, elementeMitEigenschaftM, elementeN):
    mean, var, skew, kurt = hypergeom.stats(elementeN, nStichprobe, elementeMitEigenschaftM, moments='mvsk')
    sum = 0
    for i in array:
        sum += scipy.stats.distributions.hypergeom.pmf(i, elementeN, nStichprobe, elementeMitEigenschaftM)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)
    # print("Wahrsch: Aus", nElements, "Elementen, wovon", mEigenschaft, "eine Eigenschaft haben", kZiehen, "zu ziehen: ",


def geometrisch_verteilung(p, array):
    mean, var, skew, kurt = geom.stats(p, moments='mvsk')
    sum = 0
    for i in array:
        sum += scipy.stats.distributions.geom.pmf(i, p)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)


def poisson_verteilung(array, rate):
    mean, var, skew, kurt = poisson.stats(rate, moments='mvsk')

    sum = 0
    for i in array:
        sum += scipy.stats.distributions.poisson.pmf(i, rate)
    print("Wahrscheinlichkeit", sum)
    print("Erwartungswert", mean)
    print("Varianz", var)
    print("Schiefe", skew)
    print("Wölbung", kurt)


# TESTS############################
def absoluteHaufigkeitTest():
    assert (absoluteHaufigkeit([1, 2, 2, 4, 5]) == {1: 1, 2: 2, 4: 1, 5: 1})
    assert (absoluteHaufigkeit([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3})
    assert (absoluteHaufigkeit([2, 1, 5, 3, 1, 5]) == {1: 2, 2: 1, 3: 1, 5: 2})


def relativeHaufigkeitTest():
    assert (relativeHaufigkeit([1, 2, 2, 4, 5]) == {1: 1 / 5, 2: 2 / 5, 4: 1 / 5, 5: 1 / 5})
    assert (relativeHaufigkeit([1, 2, 2, 3, 3, 3]) == {1: 1 / 6, 2: 2 / 6, 3: 3 / 6})
    assert (relativeHaufigkeit([2, 1, 5, 3, 1, 5, 6]) == {1: 2 / 7, 2: 1 / 7, 3: 1 / 7, 5: 2 / 7, 6: 1 / 7})


def empirirscheVerteilungsfunktionTest():
    assert (empirirscheVerteilungsfunktion([1, 2, 3, 4, 5]) == {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0})


def mittelwertTest():
    assert (mittelwert([1, 2, 3, 4, 5]) == 3)
    assert (mittelwert([1, 1, 1, 1, 2]) == 1.2)
    assert (mittelwert([5, 0, 295]) == 100)


def arithmMittelTest():
    assert (arithmMittel([1, 2, 3, 4, 5]) == 3)
    assert (arithmMittel([1, 1, 1, 1, 2]) == 1.2)
    assert (arithmMittel([5, 0, 295]) == 100)


def medianTest():
    assert (median([1, 2, 3, 4, 5]) == 3)
    assert (median([1, 1, 1, 1, 2]) == 1)
    assert (median([5, 0, 295]) == 5)


def modalTest():
    assert (modal([1, 2, 3, 3, 3, 4, 5]) == [3])
    assert (modal([1, 1, 1, 1, 2]) == 1)
    # todo more than 2 modalvalue gives first


def pQuantilTest():
    list = [14, 15, 17, 18, 19, 22, 24, 29, 36, 41]
    assert (pQuantil(list, 25) == 17)
    assert (pQuantil(list, 50) == 20.5)
    assert (pQuantil(list, 75) == 29)
    assert (pQuantil(list, 90) == 38.5)
    # todo wrong


def empirischNminus1VarianzTest():
    assert (empirischNminus1Varianz([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 2.8)
    # assert (empirischNminus1Varianz([-2, -1, 0, 1, 2]) == 2.5)
    # assert (empirischNminus1Varianz([-20, -10, 0, 10, 20]) == 250)


def empirischeNminus1StandardabweichungTest():
    assert (empirischeNminus1Standardabweichung([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 1.6733200530681511)
    # assert (empirischeNminus1Standardabweichung([-2, -1, 0, 1, 2]) == 1.5811388300841898)
    # assert (empirischeNminus1Standardabweichung([-20, -10, 0, 10, 20]) == 15.811388300841896)


def normaleVarianzTest():
    pass
    # assert (normaleVarianz([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 2.8)
    # assert (empirischNminus1Varianz([-2, -1, 0, 1, 2]) == 2.5)
    # assert (empirischNminus1Varianz([-20, -10, 0, 10, 20]) == 250)


def normaleStandardabweichungTest():
    pass
    # assert (normaleStandardabweichung([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 1.6733200530681511)
    # assert (empirischeNminus1Standardabweichung([-2, -1, 0, 1, 2]) == 1.5811388300841898)
    # assert (empirischeNminus1Standardabweichung([-20, -10, 0, 10, 20]) == 15.811388300841896)


def spannweiteTest():
    assert (spannweite([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 6)
    assert (spannweite([1, 1, 2, 2, 4, 4, 7, 3, 3, 3, 3]) == 6)


def interquartilabstandIRQTest():
    assert (interquartilabstandIRQ([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]))
    # todo wrong


if __name__ == '__main__':
    # absoluteHaufigkeitTest()
    # relativeHaufigkeitTest()
    # empirirscheVerteilungsfunktionTest()
    # mittelwertTest()
    # arithmMittelTest()
    # medianTest()
    # modalTest()
    pQuantilTest()
    # empirischNminus1VarianzTest()
    # empirischeNminus1StandardabweichungTest()
    # normaleVarianzTest()
    # normaleStandardabweichungTest()
    # spannweiteTest()
    # interquartilabstandIRQTest()
