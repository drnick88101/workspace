## A function that can take any non-negative integer as an argument and return it with its digits in descending order.

def Descending_Order(num):
    num = str(num)
    num = list(num)
    num.sort(reverse=True)
    num = ''.join(num)
    return int(num)