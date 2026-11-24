import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import build_hospital_graph, bfs, dfs_iterative, iddfs_recursive

g = build_hospital_graph(31, 2)
for label, start, goal in [('Best case', 0, 0), ('Goal at end', 0, 30)]:
    print(label)
    for name, fn in [('BFS', bfs), ('DFS', dfs_iterative), ('IDDFS', iddfs_recursive)]:
        r=fn(g,start,goal)
        print(f'  {name}: found={r["found"]}, generated={r["generated"]}, expanded={r["expanded"]}')
