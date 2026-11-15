import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import build_hospital_graph, bfs, dfs_iterative, iddfs_recursive, uniform_cost_search

algorithms = [('BFS', bfs), ('Iterative DFS', dfs_iterative), ('Recursive IDDFS', iddfs_recursive)]
graph = build_hospital_graph(40, 3)

print('CASE 1: reachable goal (0 to 39)')
for name, fn in algorithms:
    r=fn(graph,0,39)
    print(f'{name}: found={r["found"]}, path={r["path"]}, depth={r["depth"]}, cost={r["path_cost"]}')

print('CASE 2: start equals goal (7 to 7)')
for name, fn in algorithms:
    r=fn(graph,7,7)
    print(f'{name}: found={r["found"]}, path={r["path"]}, depth={r["depth"]}')

print('CASE 3: unreachable goal (remove all edges to node 39)')
without_goal_edges=[[(v,w) for v,w in edges if v!=39] for edges in graph]
for name, fn in algorithms:
    r=fn(without_goal_edges,0,39)
    print(f'{name}: found={r["found"]}, path={r["path"]}')

print('CASE 4: weighted route (0 to 3)')
weighted=[[(1,8),(2,2)],[(3,2)],[(3,9)],[]]
r=uniform_cost_search(weighted,0,3)
print('UCS: path=',r['path'],'travel_cost=',r['path_cost'],'optimal=',r['optimal'])
