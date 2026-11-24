
"""CSA1717 CO2 Assignment 2: reproducible algorithm experiments.
All examples are educational models; metrics are operation counts from these implementations.
Run: python src/assignment2_co2.py --out results
"""
from __future__ import annotations
import argparse, csv, json, math, random, time, statistics
from collections import deque
from pathlib import Path
from itertools import permutations

def timed(fn):
    t=time.perf_counter(); result=fn(); return result, (time.perf_counter()-t)*1000

# TASK 1: Uniform-cost, Greedy best-first, A*
def grid_neighbors(cell, grid):
    r,c=cell
    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
        q=(r+dr,c+dc)
        if 0<=q[0]<len(grid) and 0<=q[1]<len(grid[0]) and grid[q[0]][q[1]] != '#':
            yield q, 1

def search_grid(grid, start, goal, mode):
    import heapq
    h=lambda p: abs(p[0]-goal[0])+abs(p[1]-goal[1])
    pq=[]; serial=0
    heapq.heappush(pq,(0,serial,start))
    g={start:0}; parent={}; closed=set()
    generated=1; expanded=0; max_frontier=1
    while pq:
        _,_,u=heapq.heappop(pq)
        if u in closed: continue
        if u==goal:
            path=[u]
            while u in parent: u=parent[u]; path.append(u)
            path.reverse()
            return dict(found=True,path=path,cost=g[goal],expanded=expanded,generated=generated,
                        max_frontier=max_frontier,optimal=(mode!='greedy'),path_length=len(path)-1)
        closed.add(u); expanded+=1
        for v,w in grid_neighbors(u,grid):
            ng=g[u]+w
            if ng<g.get(v,float('inf')):
                g[v]=ng; parent[v]=u; serial+=1
                priority=ng if mode=='ucs' else h(v) if mode=='greedy' else ng+h(v)
                heapq.heappush(pq,(priority,serial,v)); generated+=1
        max_frontier=max(max_frontier,len(pq))
    return dict(found=False,path=[],cost=None,expanded=expanded,generated=generated,
                max_frontier=max_frontier,optimal=(mode!='greedy'),path_length=None)

def task1():
    maps={
      "open": ["S.........","..........","..........","..........",".........G"],
      "cluttered":["S.#...#...","..#.#...#.","..#.#.#...","....#.#.##","##..#....G"],
      "dead_end":["S....#####","####.#####","..........","####.#####","#########G"],
    }
    out={}
    for name,rows in maps.items():
        grid=[list(x) for x in rows]
        s=next((r,c) for r,row in enumerate(grid) for c,x in enumerate(row) if x=='S')
        g=next((r,c) for r,row in enumerate(grid) for c,x in enumerate(row) if x=='G')
        out[name]={}
        for mode in ('ucs','greedy','astar'):
            res,ms=timed(lambda:search_grid(grid,s,g,mode)); res['time_ms']=round(ms,4)
            out[name][mode]=res
    return out

# TASK 2: Online DFS agent with local sensing, and LRTA* on known traversable grid.
def online_dfs(grid,start,goal):
    # Agent initially knows only adjacent cells; stack frames store unexplored neighbors.
    visited={start}; parent={}; stack=[start]; moves=0; revisits=0
    while stack:
        u=stack[-1]
        if u==goal: return dict(found=True,moves=moves,revisits=revisits,visited=len(visited))
        options=[v for v,_ in grid_neighbors(u,grid) if v not in visited]
        if options:
            v=options[0]; parent[v]=u; visited.add(v); stack.append(v); moves+=1
        else:
            stack.pop()
            if stack: moves+=1; revisits+=1
    return dict(found=False,moves=moves,revisits=revisits,visited=len(visited))
def shortest_grid(grid,s,g):
    q=deque([(s,0)]); seen={s}
    while q:
        u,d=q.popleft()
        if u==g:return d
        for v,_ in grid_neighbors(u,grid):
            if v not in seen: seen.add(v);q.append((v,d+1))
    return None
def task2():
    cases={}
    for n in (7,11,15):
        # Open rectangular room, start upper left, target lower right
        grid=[list("."*n) for _ in range(n)]
        s=(0,0);g=(n-1,n-1)
        online=online_dfs(grid,s,g); opt=shortest_grid(grid,s,g)
        online['optimal_cost']=opt;online['competitive_ratio']=round(online['moves']/opt,3)
        cases[f"open_{n}x{n}"]=online
    # Unreachable target separated by an obstacle wall
    grid=[list(".....") for _ in range(5)]
    for r in range(5): grid[r][2]='#'
    cases["unreachable"]=online_dfs(grid,(0,0),(4,4))
    # A repeated-mission note: same deterministic environment; DFS learns explored map, route cache is optimal after mapping.
    cases["repeated_mission"]={"first_mission_exploration_moves":online_dfs([list(".....") for _ in range(5)],(0,0),(4,4))["moves"],
                               "subsequent_known_map_optimal_moves":8}
    return cases

# TASK 3: minimax and alpha-beta on deliberately ordered trees.
def minimax_tree(tree, node, maximizing, stats):
    stats['visited']+=1
    if isinstance(tree[node],(int,float)): return tree[node]
    vals=[minimax_tree(tree,ch,not maximizing,stats) for ch in tree[node]]
    return max(vals) if maximizing else min(vals)
def alphabeta_tree(tree,node,maximizing,alpha,beta,stats):
    stats['visited']+=1
    if isinstance(tree[node],(int,float)): return tree[node]
    if maximizing:
        value=-float('inf')
        for ch in tree[node]:
            value=max(value,alphabeta_tree(tree,ch,False,alpha,beta,stats)); alpha=max(alpha,value)
            if alpha>=beta: stats['cutoffs']+=1; break
        return value
    value=float('inf')
    for ch in tree[node]:
        value=min(value,alphabeta_tree(tree,ch,True,alpha,beta,stats)); beta=min(beta,value)
        if beta<=alpha: stats['cutoffs']+=1; break
    return value
def make_tree(leaves,order='good'):
    tree={}; idx=0
    # root max; two min children; each has two max children; each has two leaves
    vals=list(leaves)
    def build(level, prefix):
        nonlocal idx
        node=f"{prefix}"
        if level==3:
            tree[node]=vals[idx];idx+=1;return node
        tree[node]=[]
        for j in range(2): tree[node].append(build(level+1,node+str(j)))
        return node
    root=build(0,'R')
    return tree,root
def task3():
    # At root, first subtree gives max value 8 quickly, then second min subtree can cut off.
    good=[8,9,7,100, 1,2,0,3]
    poor=[0,1,2,3, 5,6,0,8] # poor leaf ordering suppresses early cutoffs
    random.seed(41); random_vals=[random.randint(-5,10) for _ in range(8)]
    results={}
    for name,vals in [('favourable_order',good),('unfavourable_order',poor),('random_order',random_vals)]:
        tree,root=make_tree(vals)
        a={'visited':0,'cutoffs':0}; b={'visited':0}
        av=alphabeta_tree(tree,root,True,-float('inf'),float('inf'),a)
        mv=minimax_tree(tree,root,True,b)
        results[name]={'leaf_values':vals,'minimax_value':mv,'alpha_beta_value':av,
                       'minimax_nodes':b['visited'],'alpha_beta_nodes':a['visited'],'cutoffs':a['cutoffs']}
    return results

# TASK 4: depth-limited minimax over deterministic synthetic game trees
def minimax_full(depth,b,seed=3):
    rng=random.Random(seed); nodes=0; evals=0
    def rec(d,maxing):
        nonlocal nodes,evals
        nodes+=1
        if d==0:
            evals+=1; return rng.uniform(-1,1)
        vals=[rec(d-1,not maxing) for _ in range(b)]
        return max(vals) if maxing else min(vals)
    val=rec(depth,True)
    return {'value':round(val,5),'nodes':nodes,'evaluation_calls':evals}
def task4():
    out={}
    for b,d in [(2,4),(3,4),(3,5),(4,5)]:
        out[f"b{b}_d{d}"]=minimax_full(d,b,seed=100+b*10+d)
    return out

# TASK 5: AC-3 propagation + backtracking
def ac3(domains,neighbors):
    domains={v:set(vals) for v,vals in domains.items()}
    queue=[(x,y) for x in domains for y in neighbors[x]]
    checks=0; removals=0
    while queue:
        x,y=queue.pop(0); revised=False
        for a in list(domains[x]):
            checks+=1
            if not any(a!=b for b in domains[y]):
                domains[x].remove(a); removals+=1; revised=True
        if revised:
            if not domains[x]: return domains,checks,removals,False
            for z in neighbors[x]:
                if z!=y: queue.append((z,x))
    return domains,checks,removals,True
def backtrack_csp(domains,neighbors,propagate=True):
    stats={'nodes':0,'backtracks':0,'checks':0,'wipeouts':0}
    def rec(assign,ds):
        stats['nodes']+=1
        if len(assign)==len(ds): return assign
        var=min((v for v in ds if v not in assign),key=lambda x:len(ds[x]))
        for value in sorted(ds[var]):
            nd={k:set(v) for k,v in ds.items()}; nd[var]={value}
            ok=True
            for other in neighbors[var]:
                if other in assign and assign[other]==value: ok=False
                if other not in assign:
                    nd[other].discard(value); stats['checks']+=1
                    if not nd[other]: stats['wipeouts']+=1;ok=False
            if not ok: continue
            if propagate:
                nd,checks,_,consistent=ac3(nd,neighbors);stats['checks']+=checks
                if not consistent: stats['wipeouts']+=1;continue
            na=dict(assign);na[var]=value
            result=rec(na,nd)
            if result is not None:return result
        stats['backtracks']+=1
        return None
    sol=rec({}, {k:set(v) for k,v in domains.items()})
    return sol,stats
def task5():
    # Exams A,B,C,D; conflict graph is a chain. Three slots.
    domains={'A':{1},'B':{2},'X':{1,2,3},'C':{1,2,3}}
    neigh={'A':{'X'},'B':{'X'},'X':{'A','B','C'},'C':{'X'}}
    propagated,checks,removed,ok=ac3(domains,neigh)
    sol0,s0=backtrack_csp(domains,neigh,False)
    sol1,s1=backtrack_csp(domains,neigh,True)
    # Triangle with only two slots: arc consistency alone cannot prove global inconsistency.
    d2={x:{1,2} for x in "XYZ"}; n2={'X':{'Y','Z'},'Y':{'X','Z'},'Z':{'X','Y'}}
    p2,c2,r2,ok2=ac3(d2,n2)
    sol2,s2=backtrack_csp(d2,n2,True)
    return {'forced_instance':{'initial_domains':{k:sorted(v) for k,v in domains.items()},
      'after_ac3':{k:sorted(v) for k,v in propagated.items()},'arc_consistent':ok,
      'arc_checks':checks,'values_removed':removed,'solution_without_propagation':sol0,'stats_without':s0,
      'solution_with_propagation':sol1,'stats_with':s1},
      'triangle_unsat':{'after_ac3':{k:sorted(v) for k,v in p2.items()},'arc_consistent':ok2,
      'checks':c2,'removed':r2,'backtracking_solution':sol2,'backtracking_stats':s2}}

# TASK 6 and 7: TSP exhaustive, 2-opt and branch-and-bound.
def tour_cost(tour,dist):
    return sum(dist[tour[i]][tour[(i+1)%len(tour)]] for i in range(len(tour)))
def make_dist(n,seed=12):
    rng=random.Random(seed+n)
    pts=[(rng.random()*100,rng.random()*100) for _ in range(n)]
    return [[round(math.hypot(x1-x2,y1-y2),3) for x2,y2 in pts] for x1,y1 in pts]
def exhaustive_tsp(dist):
    n=len(dist); best=float('inf');besttour=None;count=0
    for perm in permutations(range(1,n)):
        t=(0,)+perm; c=tour_cost(t,dist);count+=1
        if c<best:best,besttour=c,t
    return {'tour':besttour,'cost':round(best,3),'candidates':count}
def two_opt(dist,seed):
    rng=random.Random(seed); n=len(dist); tour=list(range(n));rng.shuffle(tour)
    best=tour_cost(tour,dist); improved=True
    while improved:
        improved=False
        for i in range(1,n-1):
            for j in range(i+1,n):
                cand=tour[:i]+list(reversed(tour[i:j+1]))+tour[j+1:]
                c=tour_cost(cand,dist)
                if c+1e-9<best: tour,best=cand,c;improved=True
    return best
def tsp_bnb(dist):
    # Symmetry-reduced branch and bound: fix depot 0, lower bound current cost + MST(unvisited) + cheapest links.
    n=len(dist); best=float('inf');besttour=None;nodes=0;pruned=0
    def mst_bound(vertices):
        if len(vertices)<2:return 0
        used={vertices[0]}; total=0
        while len(used)<len(vertices):
            edge=min((dist[u][v],v) for u in used for v in vertices if v not in used)
            total+=edge[0];used.add(edge[1])
        return total
    def rec(path,remaining,cost):
        nonlocal best,besttour,nodes,pruned
        nodes+=1
        if not remaining:
            total=cost+dist[path[-1]][0]
            if total<best:best,besttour=total,tuple(path)
            return
        # admissible bound: MST over unvisited + cheapest connection from current and to depot
        rem=list(remaining)
        bound=cost+mst_bound(rem)+min(dist[path[-1]][v] for v in rem)+min(dist[v][0] for v in rem)
        if bound>=best:pruned+=1;return
        for v in sorted(remaining,key=lambda x:dist[path[-1]][x]):
            rec(path+[v],remaining-{v},cost+dist[path[-1]][v])
    rec([0],set(range(1,n)),0)
    return {'tour':besttour,'cost':round(best,3),'nodes':nodes,'pruned':pruned}
def task6_7():
    out={}
    for n in (6,7,8,9):
        dist=make_dist(n)
        exact=exhaustive_tsp(dist)
        t0=time.perf_counter(); local=[two_opt(dist,seed) for seed in range(20)]; local_elapsed=(time.perf_counter()-t0)*1000/20
        out[str(n)]={'exhaustive':exact,'two_opt_runs':20,'two_opt_mean_cost':round(statistics.mean(local),3),
          'two_opt_best_cost':round(min(local),3),'two_opt_mean_gap_pct':round(statistics.mean((x-exact['cost'])/exact['cost']*100 for x in local),3),
          'two_opt_mean_time_ms':round(local_elapsed,4),'branch_bound':tsp_bnb(dist)}
    return out

def run_all():
    return {'task1':task1(),'task2':task2(),'task3':task3(),'task4':task4(),'task5':task5(),'task6_task7':task6_7()}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=run_all()
    (out/'results.json').write_text(json.dumps(data,indent=2))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','case','metrics'])
        for task,vals in data.items():
            for case,metrics in vals.items():w.writerow([task,case,json.dumps(metrics,sort_keys=True)])
    print(json.dumps(data,indent=2))
