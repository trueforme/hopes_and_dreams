import sys
from collections import deque, defaultdict

def is_gateway(vertex):
    return vertex and vertex[0].isupper()

def is_node(vertex):
    return vertex and vertex[0].islower()

def build_static_graph_parts(edges):
    nodes = set()
    removable_gateway_edges = set()
    node_edges = set()
    for left_node, right_node in edges:
        nodes.add(left_node)
        nodes.add(right_node)
        if is_gateway(left_node) and is_node(right_node):
            removable_gateway_edges.add((left_node, right_node))
        elif is_gateway(right_node) and is_node(left_node):
            removable_gateway_edges.add((right_node, left_node))
        else:
            node_edges.add(tuple(sorted((left_node, right_node))))
    return nodes, removable_gateway_edges, node_edges

def build_node_only_graph(nodes, node_edges):
    node_graph = defaultdict(set)
    for left_node, right_node in node_edges:
        if is_node(left_node) and is_node(right_node):
            node_graph[left_node].add(right_node)
            node_graph[right_node].add(left_node)
    if 'a' in nodes and 'a' not in node_graph:
        node_graph['a'] = set()
    return node_graph

def build_full_graph(nodes, node_edges, active_removable_gateway_edges):
    full_graph = {node: set() for node in nodes}
    for left_node, right_node in node_edges:
        full_graph[left_node].add(right_node)
        full_graph[right_node].add(left_node)
    for gateway, node in active_removable_gateway_edges:
        full_graph[gateway].add(node)
        full_graph[node].add(gateway)
    return full_graph

def preprocess_all_pairs_node_distances(node_graph):
    distances_by_start_node = {}
    for start_node in node_graph.keys():
        distance = {start_node: 0}
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            for neighbor_node in node_graph[current_node]:
                if neighbor_node not in distance:
                    distance[neighbor_node] = distance[current_node] + 1
                    queue.append(neighbor_node)
        distances_by_start_node[start_node] = distance
    return distances_by_start_node

def compute_distance_node_to_gateway(node_label, gateway_label, active_removable_gateway_edges, all_pairs_node_distances):
    best_distance = None
    for gw, adjacent_node in active_removable_gateway_edges:
        if gw != gateway_label:
            continue
        if node_label == adjacent_node:
            candidate = 1
        else:
            if node_label in all_pairs_node_distances and adjacent_node in all_pairs_node_distances[node_label]:
                candidate = all_pairs_node_distances[node_label][adjacent_node] + 1
            else:
                continue
        if best_distance is None or candidate < best_distance:
            best_distance = candidate
    return best_distance

def select_nearest_gateway_for_position(virus_position, gateways, active_removable_gateway_edges, all_pairs_node_distances):
    best_gateway = None
    best_distance = None
    for gateway in sorted(gateways):
        if is_node(virus_position):
            cur_distance = compute_distance_node_to_gateway(virus_position, gateway, active_removable_gateway_edges, all_pairs_node_distances)
        else:
            cur_distance = 0 if virus_position == gateway else None
        if cur_distance is None:
            continue
        if best_gateway is None or cur_distance < best_distance or (cur_distance == best_distance and gateway < best_gateway):
            best_gateway = gateway
            best_distance = cur_distance
    return best_gateway, best_distance

def simulate_virus_single_step(full_graph, virus_position, gateways, active_removable_gateway_edges, all_pairs_node_distances):
    gateway, distance_here = select_nearest_gateway_for_position(virus_position, gateways, active_removable_gateway_edges, all_pairs_node_distances)
    if gateway is None:
        return None
    if distance_here == 0:
        return virus_position
    candidates = []
    for neighbor in sorted(full_graph[virus_position]):
        if is_gateway(neighbor):
            neighbor_distance = 0 if neighbor == gateway else None
        else:
            neighbor_distance = compute_distance_node_to_gateway(neighbor, gateway, active_removable_gateway_edges, all_pairs_node_distances)
        if neighbor_distance is not None and neighbor_distance == distance_here - 1:
            candidates.append(neighbor)
    if not candidates:
        return None
    return candidates[0]

def solve(edges):
    nodes, removable_gateway_edges_initial, node_edges = build_static_graph_parts(edges)
    virus_start = 'a'
    if virus_start not in nodes:
        nodes.add(virus_start)
    gateways = sorted([x for x in nodes if is_gateway(x)])
    removable_gateway_edges_initial = set(removable_gateway_edges_initial)
    node_graph = build_node_only_graph(nodes, node_edges)
    all_pairs_node_distances = preprocess_all_pairs_node_distances(node_graph)
    initial_state = (virus_start, frozenset(removable_gateway_edges_initial))
    queue = deque()
    queue.append((initial_state, []))
    visited_states = {initial_state}
    while queue:
        (virus_position, active_removable_gateway_edges), actions_taken = queue.popleft()
        nearest_gateway, _ = select_nearest_gateway_for_position(virus_position, gateways, active_removable_gateway_edges, all_pairs_node_distances)
        if nearest_gateway is None:
            return actions_taken[:]
        for gateway, node in sorted(active_removable_gateway_edges):
            updated_gateway_edges = set(active_removable_gateway_edges)
            updated_gateway_edges.remove((gateway, node))
            full_graph_after_cut = build_full_graph(nodes, node_edges, updated_gateway_edges)
            nearest_after_cut, _ = select_nearest_gateway_for_position(virus_position, gateways, updated_gateway_edges, all_pairs_node_distances)
            if nearest_after_cut is None:
                return actions_taken + [f"{gateway}-{node}"]
            new_virus_position = simulate_virus_single_step(full_graph_after_cut, virus_position, gateways, updated_gateway_edges, all_pairs_node_distances)
            if new_virus_position is None:
                return actions_taken + [f"{gateway}-{node}"]
            if is_gateway(new_virus_position):
                continue
            new_state = (new_virus_position, frozenset(updated_gateway_edges))
            if new_state not in visited_states:
                visited_states.add(new_state)
                queue.append((new_state, actions_taken + [f"{gateway}-{node}"]))
    return []

def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()