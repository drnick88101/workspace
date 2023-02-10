# Larry Maes
# 09/03/2021
# S301 Assignment 1
# a simple FOR loop that adds the first 100 positive integers (from 1 to 100, included).

import random
matrix1 = [[int(random.random()*10) for x in range(5)] for y in range(5)]
matrix2 = [[int(random.random()*10) for x in range(5)] for y in range(5)]
matrix3 = [[0 for x in range(5)] for y in range(5)]
for i in range(5):
    for j in range(5):
        for k in range(5):
            total = matrix1[i][k] * matrix2[k][j]
            matrix3[i][j] = matrix3[i][j] + total

for r in matrix1:
    for c in r:
        print(c,end = " ")
    print()
print("\n")

for r in matrix2:
    for c in r:
        print(c,end = " ")
    print()
print("\n")

for r in matrix3:
    for c in r:
        print(c,end = " ")
    print()