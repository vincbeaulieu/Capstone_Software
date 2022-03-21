
from itertools import combinations


def group_gen(gestures, group_size, group_qty):
    print("Gestures: ", gestures)

    gestures_groups = []
    tmp_list = []
    group_nb, i = 0, 1
    while group_nb <= group_qty:
        gesture = gestures[(i-1) % len(gestures)]

        if i % group_size == 0:
            ((i - 1) != 0) and gestures_groups.append(tmp_list)
            tmp_list = [gesture]
            group_nb += 1
        else:
            tmp_list.append(gesture)

        i += 1
    return gestures_groups[1:group_qty+1]


def powerset(iterable, subset_size):
    return list(combinations(iterable, subset_size))


def factors(number):
    return list(i for i in range(1, number + 1) if number % i == 0)


