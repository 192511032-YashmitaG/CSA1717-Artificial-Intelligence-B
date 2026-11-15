import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
from pathlib import Path
from search_algorithms import build_hospital_graph, bfs, dfs_iterative, iddfs_recursive

sizes=[1_000,10_000,100_000,1_000_000]
algorithms=[('BFS',bfs),('Iterative DFS',dfs_iterative),('Recursive IDDFS',iddfs_recursive)]
rows=[]
for n in sizes:
    graph=build_hospital_graph(n,3)
    for name,fn in algorithms:
        runs=[fn(graph,0,n-1) for _ in range(5)]
        last=runs[-1]
        rows.append({'states':n,'algorithm':name,'generated':last['generated'],
                     'expanded':last['expanded'],'max_frontier':last['max_frontier'],
                     'depth':last['depth'],'path_cost':last['path_cost'],
                     'optimal':last['optimal'],
                     'avg_time_ms':sum(x['time_ms'] for x in runs)/len(runs),'runs':5})
    print(f'Completed {n:,} states')

out=Path(__file__).resolve().parent/'performance_results.txt'
with out.open('w',encoding='utf-8') as f:
    f.write('SEED: deterministic graph generator (no random values)\n')
    f.write('RUNS PER ALGORITHM AND SIZE: 5\n')
    f.write('states | algorithm | generated | expanded | max_frontier | depth | path_cost | optimal | avg_time_ms\n')
    for r in rows:
        f.write(f"{r['states']} | {r['algorithm']} | {r['generated']} | {r['expanded']} | {r['max_frontier']} | {r['depth']} | {r['path_cost']} | {r['optimal']} | {r['avg_time_ms']:.3f}\n")
print('Saved',out)
