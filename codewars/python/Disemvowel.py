## A function that takes a string and returns a new string with all vowels removed.

def disemvowel(string_):
    vowels = "aeiouAEIOU"
    new_string = ""
    for letter in string_:
        if letter not in vowels:
            new_string += letter
    return new_string