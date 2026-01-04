def build_supragraph(G_dict, omega=1.0, normalize_weights=False, adaptive_omega=True):
    """
    Build a supra-graph connecting temporal layers
    
    Parameters:
    -----------
    G_dict: dict
        Dictionary of graphs {year: Graph}
    omega: float
        Inter-layer coupling strength (used if adaptive_omega=False)
    normalize_weights: bool
        If True, normalize intra-layer weights to [0, 1] range
    adaptive_omega: bool
        If True, use adaptive omega: 1.0 for years ≤2001, mean weight for years >2001
    
    Returns:
    --------
    G_supra: nx.Graph
        Supra-graph with nodes labeled as (node, year)
    """
    import networkx as nx
    import numpy as np
    
    G_supra = nx.Graph()
    
    # Normalization constant
    max_weight = 189.0 if normalize_weights else 1.0  # Preset max weight based on data knowledge
    
    # Add intra-layer edges (within each year) - vectorized
    for year, G in G_dict.items():
        # Add all nodes for this year (even if isolated) - numpy style
        nodes_array = np.array(list(G.nodes()))
        year_array = np.full(len(nodes_array), year)
        supra_nodes_matrix = np.column_stack([nodes_array, year_array])  # Nx2 matrix

        # Convert rows to tuples using map
        supra_nodes = list(map(tuple, supra_nodes_matrix))
        G_supra.add_nodes_from(supra_nodes)    #168 nodes added in the format (family, 'year')

        edges_array = np.array(list(G.edges(data='weight')))
        if len(edges_array) > 0:
            
            # Normalize weights
            weights = edges_array[:, 2].astype(int) / max_weight
            
            # Add all edges at once
            u_nodes = edges_array[:, 0]
            v_nodes = edges_array[:, 1]
            
            # Use zip to create edges - faster than indexing
            edges_to_add = [
                ((u, str(year)), (v, str(year)), {'weight': w, 'edge_type': 'intra'})
                for u, v, w in zip(u_nodes, v_nodes, weights)
            ]
            
            G_supra.add_edges_from(edges_to_add)
    
    # Add inter-layer edges (same node across consecutive years)
    years = sorted(G_dict.keys())
    families = list(G_dict[years[0]].nodes())  # All families (constant across years)
    
    # Calculate adaptive omega values if needed
    if adaptive_omega:
        omega_dict = {}
        for year in years:
            G = G_dict[year]
            if G.number_of_edges() > 0:
                # Use omega=1 for years ≤2001, mean weight for years >2001
                if year <= 2001:
                    omega_dict[year] = 1.0
                else:
                    weights = np.array(list(G.edges(data='weight')))[:,2].astype(int)
                    mean_weight = np.mean(weights)
                    omega_dict[year] = mean_weight
            else:
                omega_dict[year] = 1.0  # Default for years with no edges
    
    # Add inter-layer edges year by year
    for i, year in enumerate(years[:-1]): #iterate over all years until 2009
        next_year = years[i + 1]
        
        # Determine omega for this transition
        if adaptive_omega:
            # Use omega from current year for the transition to next year
            year_omega = omega_dict[year]
        else:
            year_omega = omega
        
        # Add edges for all families between this year and next
        inter_edges = [
            ((family, str(year)), (family, str(next_year)), {'weight': year_omega, 'edge_type': 'inter'})
            for family in families
        ]
        
        G_supra.add_edges_from(inter_edges)
    
    return G_supra