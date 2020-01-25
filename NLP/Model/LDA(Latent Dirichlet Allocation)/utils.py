import random


"""
Choose a element in @vec according to a specified distribution @pr
Reference:
http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
"""


def choose(target_list, probability):
    assert len(target_list) == len(probability)
    p_s = sum(probability)
    probability = [float(p)/p_s for p in probability]
    target_index = -1
    r = random.random()
    while r > 0:
        r = r - probability[target_index]
        target_index += 1
    return target_index
