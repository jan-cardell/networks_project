def merge_fams(binary_adj):
    import pandas as pd
    original_df = pd.read_csv(binary_adj)

    sinaloa = original_df['Sinaloa'] + original_df['Sinaloa_Family']
    sinaloa = (sinaloa > 0).astype(int)

    beltran_leyva = original_df['Beltran_Leyva'] + original_df['Beltran_Leyva_Family']
    beltran_leyva = (beltran_leyva > 0).astype(int)

    cleaned_df = original_df.copy()
    cleaned_df.drop(columns={'Sinaloa_Family', 'Beltran_Leyva_Family'}, inplace=True)

    cleaned_df['Sinaloa'] = sinaloa
    cleaned_df['Beltran_Leyva'] = beltran_leyva 

    return cleaned_df
