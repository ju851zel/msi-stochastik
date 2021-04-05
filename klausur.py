# Kernel Density Estimation, KDE
import numpy as np
import scipy
#boxplot
import seaborn as seaborn


def mittelwert(array):
    return np.mean(array)


def median(array):
    return np.median(array)


def modal(array):
    # haufigste auftretende Wert
    return scipy.stats.mode(array).mode


def pQuartil(array, p):
    return np.percentile(array, p)


def varianz(array):
    # n-1
    return np.var(array, ddof=1)


def standardabweichung(array):
    # n-1
    return np.std(array, ddof=1)


def spannweite(array):
    return max(array) - min(array)


def interquartilabstandIRQ(array):
    return scipy.stats.iqr(array)


#############
def mittelwertTest():
    assert (mittelwert([1, 2, 3, 4, 5]) == 3)
    assert (mittelwert([1, 1, 1, 1, 2]) == 1.2)
    assert (mittelwert([5, 0, 295]) == 100)


def medianTest():
    assert (median([1, 2, 3, 4, 5]) == 3)
    assert (median([1, 1, 1, 1, 2]) == 1)
    assert (median([5, 0, 295]) == 5)


def modalTest():
    assert (modal([1, 2, 3, 3, 3, 4, 5]) == [3])
    assert (modal([1, 1, 1, 1, 2]) == 1)
    # todo more than 2 modalvalue gives first


def pQuartilTest():
    assert (pQuartil([14, 15, 17, 18, 19, 22, 24, 29, 36, 41], 25))
    assert (pQuartil([14, 15, 17, 18, 19, 22, 24, 29, 36, 41], 50))
    assert (pQuartil([14, 15, 17, 18, 19, 22, 24, 29, 36, 41], 75))
    assert (pQuartil([14, 15, 17, 18, 19, 22, 24, 29, 36, 41], 90))
    # todo wrong


def varianzTest():
    assert (varianz([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 2.8)
    assert (varianz([-2, -1, 0, 1, 2]) == 2.5)
    assert (varianz([-20, -10, 0, 10, 20]) == 250)


def standardabweichungTest():
    assert (standardabweichung([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 1.6733200530681511)
    assert (standardabweichung([-2, -1, 0, 1, 2]) == 1.5811388300841898)
    assert (standardabweichung([-20, -10, 0, 10, 20]) == 15.811388300841896)


def spannweiteTest():
    assert (spannweite([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]) == 6)
    assert (spannweite([1, 1, 2, 2, 4, 4, 7, 3, 3, 3, 3]) == 6)


def interquartilabstandIRQTest():
    assert (interquartilabstandIRQ([1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 7]))
    #todo wrong


if __name__ == '__main__':
    mittelwertTest()
    medianTest()
    modalTest()
    pQuartilTest()
    varianzTest()
    spannweiteTest()
    interquartilabstandIRQTest()
