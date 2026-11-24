from collections import deque
from heapq import heappush, heappop
from time import perf_counter


def build_hospital_graph(n, branching=3):
    """Create a reproducible directed tree with n locations and unit travel costs."""
    if n < 1 or branching < 1:
        raise ValueError("n and branching must be positive")
    graph = [[] for _ in range(n)]
    for u in range(n):
        first = branching * u + 1
        for v in range(first, min(first + branching, n)):
            graph[u].append((v, 1))
    return graph


def path_from_parent(parent, goal):
    path=[]; node=goal
    while node is not None:
        path.append(node); node=parent[node]
    return list(reversed(path))


def pack(path, generated, expanded, frontier, started, optimal=None):
    return {"found": path is not None, "path": path,
            "generated": generated, "expanded": expanded,
            "max_frontier": frontier,
            "depth": len(path)-1 if path else None,
            "path_cost": len(path)-1 if path else None,
            "optimal": optimal if path is not None else None,
            "time_ms": (perf_counter()-started)*1000}


def bfs(graph, start, goal):
    """Breadth-first graph search; shortest path when all edges have equal cost."""
    t=perf_counter(); q=deque([start]); parent={start:None}
    generated=1; expanded=0; maximum=1
    while q:
        u=q.popleft()
        if u==goal: return pack(path_from_parent(parent,goal),generated,expanded,maximum,t,True)
        expanded+=1
        for v,_ in graph[u]:
            if v not in parent:
                parent[v]=u; q.append(v); generated+=1
        maximum=max(maximum,len(q))
    return pack(None,generated,expanded,maximum,t)


def dfs_iterative(graph, start, goal):
    """Depth-first graph search using an explicit stack; not generally optimal."""
    t=perf_counter(); stack=[start]; parent={start:None}
    generated=1; expanded=0; maximum=1
    while stack:
        u=stack.pop()
        if u==goal: return pack(path_from_parent(parent,goal),generated,expanded,maximum,t,None)
        expanded+=1
        for v,_ in reversed(graph[u]):
            if v not in parent:
                parent[v]=u; stack.append(v); generated+=1
        maximum=max(maximum,len(stack))
    return pack(None,generated,expanded,maximum,t)


def recursive_dls(graph, u, goal, limit, path, on_path, stats):
    stats['generated']+=1
    if u==goal: return path.copy()
    if limit==0: return None
    stats['expanded']+=1
    for v,_ in graph[u]:
        if v not in on_path:
            on_path.add(v); path.append(v)
            stats['frontier']=max(stats['frontier'],len(path))
            found=recursive_dls(graph,v,goal,limit-1,path,on_path,stats)
            if found is not None: return found
            path.pop(); on_path.remove(v)
    return None


def iddfs_recursive(graph, start, goal):
    """Iterative deepening with recursive depth-limited search."""
    t=perf_counter(); total_g=0; total_e=0; maximum=1
    for limit in range(len(graph)):
        stats={'generated':0,'expanded':0,'frontier':1}
        found=recursive_dls(graph,start,goal,limit,[start],{start},stats)
        total_g+=stats['generated']; total_e+=stats['expanded']
        maximum=max(maximum,stats['frontier'])
        if found is not None:
            return pack(found,total_g,total_e,maximum,t,True)
    return pack(None,total_g,total_e,maximum,t)


def uniform_cost_search(graph, start, goal):
    """Uniform-cost search; returns a minimum-travel-cost route for non-negative costs."""
    t=perf_counter(); heap=[(0,start)]; cost={start:0}; parent={start:None}
    generated=1; expanded=0; maximum=1
    while heap:
        g,u=heappop(heap)
        if g!=cost[u]: continue
        if u==goal: return pack(path_from_parent(parent,goal),generated,expanded,maximum,t,True) | {'path_cost':g}
        expanded+=1
        for v,w in graph[u]:
            if w<0: raise ValueError('Uniform-cost search requires non-negative edge costs')
            ng=g+w
            if ng<cost.get(v,float('inf')):
                cost[v]=ng; parent[v]=u; heappush(heap,(ng,v)); generated+=1
        maximum=max(maximum,len(heap))
    return pack(None,generated,expanded,maximum,t)
