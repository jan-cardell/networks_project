import numpy as np 
import pandas as pd

def project(bipartite_df, on_municipalities=True):
    """
    Project a bipartite network (municipalities × families) onto one node type.
    
    Creates a weighted adjacency matrix where edge weights represent the number
    of shared connections between nodes in the projection.
    
    Parameters
    ----------
    bipartite_df : pd.DataFrame
        Bipartite incidence matrix with structure:
        - Index: Municipality codes (or use 'Code' column)
        - Columns: First 3 columns are metadata (Year, State, Code)
        - Remaining columns: Family names (binary 0/1 indicating presence)
    
    on_municipalities : bool, default=True
        Projection type:
        - True: Project onto municipalities (B @ B.T)
          Edge weight = number of families shared by two municipalities
        - False: Project onto families (B.T @ B)
          Edge weight = number of municipalities shared by two families
    
    Returns
    -------
    pd.DataFrame
        Weighted adjacency matrix with:
        - Rows/Columns: Municipality codes (if on_municipalities=True) 
                       or family names (if on_municipalities=False)
        - Values: Number of shared connections (weights)
        - Diagonal: Set to 0 (no self-loops)
    
    Notes
    -----
    - The projection uses matrix multiplication: A = B @ B.T (or B.T @ B)
    - Resulting matrix is symmetric and has zero diagonal
    - For year-specific projections, filter bipartite_df first:
      df_year = df[df['Year'] == 2010]
      adj = project(df_year, on_municipalities=False)
    
    See Also
    --------
    nc_backbone : Extract statistically significant backbone from projection
    """
    family_cols = bipartite_df.columns[3:]
    if bipartite_df.index.name == 'Code':
      B = bipartite_df[family_cols] 
    else:
      B = bipartite_df.set_index('Code')[family_cols]

    if on_municipalities:
        A = B @ B.T
    else:
        A = B.T @ B

    np.fill_diagonal(A.values, 0)

    return A