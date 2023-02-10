##  A function that finds the sum of the minimum values in each row of a given 2D ( nested ) list ( array, vector, .. ).

def sum_of_minimums(numbers):
    return sum([min(x) for x in numbers])
