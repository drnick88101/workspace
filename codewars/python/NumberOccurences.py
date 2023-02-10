## A function that returns the number of occurrences of an element in an array.

def number_of_occurrences(element, sample):
    total = 0
    for i in sample:
        if element == i:
            total += 1
    return total