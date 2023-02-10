# Larry Maes
# 09/03/2021
# S301 Assignment 1
# a simple FOR loop that adds the first 100 positive integers (from 1 to 100, included).

sum = 0                   # set the sum to 0

for i in range(1, 101):   # for loop to add i to sum
    sum = sum + i

print("The sum of the numbers is: ", sum)