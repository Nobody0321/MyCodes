import os


def parseData(corpusDir):
    lines = open(corpusDir).readlines()
    lines = [line.strip().split('\t') for line in lines]
    return lines


if __name__ == "__main__":

    corpusFile = "./data/corpus.txt"
    # print(parseData(corpusDir))
    lines = parseData(corpusFile)