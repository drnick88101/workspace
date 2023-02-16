## A function that takes in a string of one or more words, and returns the same string, but with all five or more letter words reversed.

def spin_words(sentence):
    output = []
    for word in sentence.split(' '):
        if len(word) > 4:
            word = word[::-1]
        output.append(word)
    return ' '.join(output)