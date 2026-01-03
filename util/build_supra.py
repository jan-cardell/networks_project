def build_supragraph(G_dict, omega=1.0, normalize_weights=True):
    """
    Build a supra-graph connecting temporal layers
    
    Parameters:
    -----------
    G_dict: dict
        Dictionary of graphs {year: Graph}
    omega: float
        Inter-layer coupling strength (relative to normalized intra-layer weights)
    normalize_weights: bool
        If True, normalize intra-layer weights to [0, 1] range
    
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
        G_supra.add_nodes_from(supra_nodes)    

        edges_array = np.array(list(G.edges(data='weight')))
        if len(edges_array) > 0:
            # Create supra-nodes: (family, year)
            u_supra = list(map(tuple, np.column_stack([edges_array[:, 0], np.full(len(edges_array), year)])))
            v_supra = list(map(tuple, np.column_stack([edges_array[:, 1], np.full(len(edges_array), year)])))
            
            # Normalize weights
            weights = edges_array[:, 2].astype(int) / max_weight
            
            # Add all edges at once
            u_nodes = edges_array[:, 0]
            v_nodes = edges_array[:, 1]
            
            # Use zip to create edges - faster than indexing
            edges_to_add = [
                ((u, year), (v, year), {'weight': w, 'edge_type': 'intra'})
                for u, v, w in zip(u_nodes, v_nodes, weights)
            ]
            
            G_supra.add_edges_from(edges_to_add)
    
    # Add inter-layer edges (same node across consecutive years)
    # unfinished
    years = sorted(G_dict.keys())
    families = G_dict[1990].nodes()

    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i+1]
        
        nodes_year1 = set(G_dict[year1].nodes())
        nodes_year2 = set(G_dict[year2].nodes())
        common_nodes = nodes_year1.intersection(nodes_year2)
        
        # Add all inter-layer edges at once
        inter_edges = [((node, year1), (node, year2), {'weight': omega, 'edge_type': 'inter'}) 
                       for node in common_nodes]
        G_supra.add_edges_from(inter_edges)
    
    return G_supra