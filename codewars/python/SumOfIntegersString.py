## A function that calculates the sum of the integers inside a string. For example, in the string "The30quick20brown10f0x1203jumps914ov3r1349the102l4zy dog", the sum of the integers is 3635.

def sum_of_integers_in_string(s):
    sum = 0
    for letter in s:
        if letter.isdigit():
            sum += int(letter)
    return sum