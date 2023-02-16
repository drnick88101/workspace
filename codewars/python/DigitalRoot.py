## A function that when given N, returns the sum of the digits of N. 

def digital_root(n):
    while n > 9:
        n = sum(int(i) for i in str(n))
    return n