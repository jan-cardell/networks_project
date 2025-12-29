def nc_backbone(adj_matrix, alpha):
    """
    Compute the noise-corrected backbone of a weighted adjacency matrix.
    
    Uses binomial test: compares observed edge weight against null model
    where probability of success of edge (i,j) → p_ij = (s_i * s_j) / W^2
    
    Parameters:
    adj_matrix (pd.DataFrame): Weighted adjacency matrix of the network.
    alpha (float): Significance level for filtering edges.

    Returns:
    np.ndarray: Edge list with shape (E, 3) - [source, target, weight]
    """
    import numpy as np
    from scipy import stats
    from util.get_edge_list import get_edge_list

    # Calculate node strengths
    strengths = adj_matrix.sum(axis=1)

    # Total network weight (sum of all edges once)
    W = adj_matrix.values.sum() / 2 # assuming undirected adjacency matrix

    if W == 0:
        # Empty graph - return empty array with proper shape
        return np.empty((0, 3), dtype=object)

    row_labels, col_labels, weights = get_edge_list(adj_matrix).T

    weights = weights.astype(int)
    
    # Get strengths using labels 
    s_i = strengths.loc[row_labels].values
    s_j = strengths.loc[col_labels].values
    
    # Calculate null probability: p_ij = (s_i * s_j) / W^2
    p_null = (s_i * s_j) / (W ** 2)
    p_null = np.clip(p_null, 0.0, 1.0)
    
    # Binomial test: P(X >= observed_weight | n=W, p=p_null)
    # Use survival function (sf) which is 1 - cdf(k-1) = P(X >= k)
    p_values = stats.binom.sf(weights - 1, n=W, p=p_null)
    
    # Filter significant edges
    significant = p_values < alpha
    
    # Return stacked edge list: shape (E, 3) - [source, target, weight]
    edge_list = np.column_stack((
        row_labels[significant],
        col_labels[significant],
        weights[significant]
    ))
    
    return edge_list