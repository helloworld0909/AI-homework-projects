import random


def weightedRandomChoice(prob):
    rand = random.random()
    for idx in range(len(prob)):
        rand -= prob[idx]
        if rand < 0:
            return idx