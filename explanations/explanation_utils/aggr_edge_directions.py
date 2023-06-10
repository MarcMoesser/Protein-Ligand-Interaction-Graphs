from collections import defaultdict

def aggregate_edge_directions(graph, edge_masks_mean):
    '''Aggregate (add) edge masks between each node pair connected by 
        edges.''' 
    edge_masks_dict = defaultdict(float)
    for val, u, v in list(zip(edge_masks_mean, *graph.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_masks_dict[(u, v)] += val
    return edge_masks_dict