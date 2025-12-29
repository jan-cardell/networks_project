def get_edge_list(adj_matrix):
    """
    Convert a weighted adjacency matrix to an edge list.
    
    Parameters:
    adj_matrix (pd.DataFrame): Weighted adjacency matrix.
    
    Returns:
    np.ndarray: Edge list with columns [node_i, node_j, weight].
    """
    import numpy as np

    A = adj_matrix.values  #numpy
    labels = adj_matrix.index.to_numpy()  # node labels
    
    # indices of upper triangle, excluding diagonal
    rows, cols = np.triu_indices(A.shape[0], k=1) #pairs of indices

    # weights at those positions
    weights = A[rows, cols]

    # keep only existing edges
    mask = weights > 0

    edge_list = np.column_stack((
        labels[rows[mask]],
        labels[cols[mask]],
        weights[mask]
    ))

    return edge_list