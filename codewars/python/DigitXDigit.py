## A function that squares every digit of a number and concatenates them.

def square_digits(num):
    num = str(num)
    new_num = ""
    for digit in num:
        new_num += str(int(digit) ** 2)
    return int(new_num)