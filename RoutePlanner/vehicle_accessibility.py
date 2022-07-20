"""
    TODO
"""

def remove_nodes(neighbour_graph, inaccessible_nodes):
    """
        neighbour_graph -> a dictionary containing indexes of cellboxes
        and how they are connected

        {
            'index':{
                '1':[index,...],
                '2':[index,...],
                '3':[index,...],
                '4':[index,...],
                '-1':[index,...],
                '-2':[index,...],
                '-3':[index,...],
                '-4':[index,...]
            },
            'index':{...},
            ...
        }

        inaccessible_nodes -> a list in indexes to be removed from the
        neighbour_graph
    """
    accessibility_graph = neighbour_graph.copy()

    for node in accessibility_graph.keys():
        for case in accessibility_graph[node].keys():
            for inaccessible_node in inaccessible_nodes:
                if int(inaccessible_node) in accessibility_graph[node][case]:
                    accessibility_graph[node][case].remove(int(inaccessible_node))

    for node in inaccessible_nodes:
        accessibility_graph.pop(int(node))

    return accessibility_graph
