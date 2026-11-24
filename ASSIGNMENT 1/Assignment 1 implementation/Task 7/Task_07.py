def T_iter(limit, b):
    values=[1]
    for _ in range(limit): values.append(1+b*values[-1])
    return values

b, d = 3, 4
vals=T_iter(d,b)
print('T(0..4):', vals)
print('Closed form T(4):', (b**(d+1)-1)//(b-1))
print('IDDFS total calls:', sum(vals))
