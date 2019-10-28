from datetime import datetime
import os

from fpgrowth import FPGrowth


def data_reader(data_file):
    data_set = []
    with open(data_file, 'r') as f:
        for line in f:
            data_set.append(line.split()[:])
    return data_set


def test_fpgrowth(data_set, min_sup=0.05):
    start = datetime.now()
    fp = FPGrowth(data_set, min_sup=min_sup)
    fp.build_fptree()
    deltatime = datetime.now() - start
    print("FP-Growth over")
    print("# of freq itemsets:", len(fp.freq_itemsets))

    return deltatime.seconds + deltatime.microseconds / 1000000
    

def run_fpgrowth():
    data_file = './../data/chess.dat'
    data_set = data_reader(data_file)
    print("FP-Growth-----------------------")
    print("Time (s):", test_fpgrowth(data_set))


if __name__ == "__main__":
    run_fpgrowth()