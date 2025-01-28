
def mean(col):
    sum = 0
    lenght = 0
    for line in col:
        sum += line
        lenght += 1
    return sum / lenght

def quartiles(col):
    lenght = 0
    for line in col:
        lenght += 1
    quartile_size = lenght / 4
    return {float(col[quartile_size]), float(col[lenght / 2]), float(col[lenght / 2 + quartile_size])}
