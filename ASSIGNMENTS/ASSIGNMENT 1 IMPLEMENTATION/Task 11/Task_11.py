import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_algorithms import uniform_cost_search
# The direct route costs 11; the two-edge route costs 10.
graph=[[(1,8),(2,2)],[(3,2)],[(3,9)],[]]
r=uniform_cost_search(graph,0,3)
print('Recommended weighted search: Uniform-Cost Search')
print('Selected path:',r['path'])
print('Total travel time:',r['path_cost'])
print('Optimal:',r['optimal'])
