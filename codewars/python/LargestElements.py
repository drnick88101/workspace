## A function that outputs the top n elements from a list

def largest(n,xs):
  list = sorted(xs)
  return list[-n:]