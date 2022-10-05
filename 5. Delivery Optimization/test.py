tuples = []

for x in range(10):
    for y in range(5):
        tuples.append((x,y))

for x,y in range(10),range(5):
    tuples.append((x,y))
print(tuples)
