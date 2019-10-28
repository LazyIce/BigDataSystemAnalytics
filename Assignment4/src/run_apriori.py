from datetime import datetime
import os
import sys

from apriori import Apriori


def data_reader(data_file):
    data_set = []
    with open(data_file, 'r') as f:
        for line in f:
            data_set.append(line.split()[:])
    return data_set


def test_apriori(data_set, min_sup):
    start = datetime.now()
    apriori = Apriori(data_set)
    apriori.generate_L(min_sup=min_sup)
    deltatime = datetime.now() - start
    print("Apriori over")
    return deltatime.seconds + deltatime.microseconds / 1000000
    print("# of freq itemsets:", len(apriori.freq_itemsets))
    print(apriori.freq_itemsets)


def run_apriori():
    data_file = sys.argv[1]
    min_sup = int(sys.argv[2])
    out_file = sys.argv[3]
    data_set = data_reader(data_file)
    print("Apriori-----------------------")
    print("Time (s):", test_apriori(data_set, min_sup))


if __name__ == "__main__":
    run_apriori()