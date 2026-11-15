from collections import deque

def no_sensor_plan(k):
    # In this simple corridor model, one MOVE_LEFT action shifts every possible
    # location one step toward the pharmacy (node 0).
    start=frozenset(range(k)); queue=deque([(start,0)]); seen={start}
    while queue:
        belief,depth=queue.popleft()
        if belief==frozenset({0}):
            return len(seen),depth
        next_belief=frozenset(max(0,x-1) for x in belief)
        if next_belief not in seen:
            seen.add(next_belief); queue.append((next_belief,depth+1))
    return len(seen),None

def local_sensor_plan(k):
    # Sensor identifies the actual junction; execute a conditional action count.
    return k, k-1

if __name__=='__main__':
    for k in (2,4,8):
        generated,steps=no_sensor_plan(k)
        sensed, worst=local_sensor_plan(k)
        print(f'k={k}: no sensor belief states={generated}, plan length={steps}; '
              f'local sensor hypotheses={sensed}, worst-case moves={worst}')
