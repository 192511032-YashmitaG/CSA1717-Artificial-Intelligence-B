import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import build_hospital_graph, bfs, dfs_iterative, iddfs_recursive

graph = build_hospital_graph(31, 2)
for name, fn in [('BFS', bfs), ('DFS', dfs_iterative), ('IDDFS', iddfs_recursive)]:
    r = fn(graph, 0, 30)
    print(f'{name}: generated={r["generated"]}, expanded={r["expanded"]}, frontier={r["max_frontier"]}')
