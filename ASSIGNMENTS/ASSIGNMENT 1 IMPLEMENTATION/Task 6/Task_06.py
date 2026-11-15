def T(limit, branching):
    if limit == 0:
        return 1
    return 1 + branching * T(limit - 1, branching)

b, limit = 3, 4
print(f'T(0) = {T(0,b)}')
for d in range(1, limit + 1):
    print(f'T({d}) = 1 + {b}*T({d-1}) = {T(d,b)}')
