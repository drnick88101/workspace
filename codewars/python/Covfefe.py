## A function that takes a string and replaces the word "coverage" with "covfefe", but only if the word "coverage" appears in the string.

def covfefe(s):
    if "coverage" in s:
        return s.replace("coverage", "covfefe")
    else:
        return s + " covfefe"
    
