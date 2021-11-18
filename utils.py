def ordered_edge(n1 : int, n2 : int):
    if n1 < n2:
        return (n1,n2)
    else:
        return (n2,n1)


def ordered_edge_weighted(n1 : int, n2 : int, w : int):
    if n1 < n2:
        return (n1,n2,w)
    else:
        return (n2,n1,w)
