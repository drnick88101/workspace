# frozen_string_literal: true

# Larry Maes
# 09/03/2021
# S301 Assignment 1
# a simple FOR loop that adds the first 100 positive integers (from 1 to 100, included).

sum = 0                 # sets the sum to 0

for i in (1..100) do    # for loop to add i to sum
  sum = sum + i
end

puts "The sum of the numbers is: #{sum}"
