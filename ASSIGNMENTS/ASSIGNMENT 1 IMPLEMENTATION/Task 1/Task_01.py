import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import build_hospital_graph, bfs, dfs_iterative, iddfs_recursive

graph = build_hospital_graph(15, branching=2)
for name, algorithm in [('BFS', bfs), ('Iterative DFS', dfs_iterative), ('Recursive IDDFS', iddfs_recursive)]:
    result = algorithm(graph, 0, 14)
    print(name, 'found =', result['found'], 'path =', result['path'])
