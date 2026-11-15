def levels(b, d):
    return [b ** i for i in range(d + 1)]

b, d = 3, 4
counts = levels(b, d)
iddfs = sum((d - i + 1) * (b ** i) for i in range(d + 1))
print('Branching factor:', b, 'Goal depth:', d)
print('Nodes at levels 0..d:', counts)
print('BFS nodes through goal level:', sum(counts))
print('IDDFS recursive calls across limits:', iddfs)
