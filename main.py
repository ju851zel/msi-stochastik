import klausur

if __name__ == '__main__':
    # klausur.hyper_geometrisch_verteilung(array=[0], elementeN=60, elementeMitEigenschaftM=10, nStichprobe=3)
    # klausur.bin_vert(4, 0.05, [0])
    # klausur.pois_vert(3.5, [0])
    # klausur.empirischeKorrellationsKoeffizient([1, 2, 3, 0, 0, 0], [0, 2, 3, 4, 5, 6])
    # klausur.hyper_geometrisch_verteilung([0], 10, 20, 120)
    # klausur.relHauf(val)
    # klausur.empirischeKorrellationsKoeffizient([2.0, 3.0, 1.0, 2.0, 1.0, 2.7],val)
    attr = [8, 7, 5, 10, 6, 3, 9, 7]
    gehalt = [55, 60, 40, 70, 45, 40, 65, 55]
    inter = [3, 10, 9, 1, 5, 10, 2, 3]
    klausur.empirischeKorrellationsKoeffizient(attr,gehalt)
