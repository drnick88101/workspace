## A function, maskify, which changes all but the last four characters of a string into '#'.

def maskify(cc):
    if len(cc) <= 4:
        return cc
    else:
        return "#"*(len(cc)-4)+cc[-4:]