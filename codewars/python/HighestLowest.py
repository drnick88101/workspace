## A function that returns the highest and lowest number in a string of space-separated numbers.

def high_and_low(numbers):
    high = low = int(numbers.split()[0])
    for num in numbers.split():
        num = int(num)
        if num > high:
            high = num
        if num < low:
            low = num
    return str(high) + ' ' + str(low)

                  