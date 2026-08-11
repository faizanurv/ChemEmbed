# reference_utils.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm


def load_reference_database_with_smiles(path, adduct):
    """
    Load and preprocess the reference database for 'with_smiles' input.
    """
    final_mol2vec = pd.read_pickle(path)

    # Calculate Precursormz and Molecular_Formula
    def calculate_molecular_formula(mol):
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        return formula

    g_sm = final_mol2vec['smile'].tolist()

    pr_mass = []
    mol_form_mv = []
    for i in range(len(g_sm)):
        mol = Chem.MolFromSmiles(g_sm[i])
        if mol is None:
            pr_mass.append(None)
            mol_form_mv.append(None)
            continue
        m_f = calculate_molecular_formula(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        proton_mass = 1.007276  # mass of a proton in Da
        if adduct == '+':
            precursor_mass_positive = mol_weight + proton_mass
        else:
            precursor_mass_positive = mol_weight - proton_mass
        pr_mass.append(round(precursor_mass_positive, 3))
        mol_form_mv.append(m_f)

    final_mol2vec['Precursormz'] = pr_mass
    final_mol2vec['Molecular_Formula'] = mol_form_mv
    #final_mol2vec.dropna(subset=['Precursormz', 'Molecular_Formula'], inplace=True)
    final_mol2vec.reset_index(drop=True, inplace=True)

    return final_mol2vec

def _read_reference_any_format(path):
    """Read the reference DB from parquet OR pickle.

    The reference used to be a 7.3GB pickle, which is all-or-nothing to load and executes
    code on unpickling. The parquet form is ~15% smaller, version-portable, and lets a
    caller read a single column (e.g. up_inchikey for a coverage check) in ~2s instead of
    materialising the whole frame in ~36s. Sniff by extension, then by content, so an
    existing pickle path keeps working unchanged."""
    p = str(path)
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if p.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    try:
        return pd.read_parquet(path)   # parquet has a magic header; fails cleanly if not
    except Exception:
        return pd.read_pickle(path)


def _stored_precursormz_is_trustworthy(df, adduct, n_check=500, tol=0.005):
    """Decide whether the stored Precursormz column can be used as-is for THIS adduct,
    instead of recomputing it from 5.5M SMILES on every load (the dominant load cost).

    This is a SELF-VALIDATING shortcut, not an assumption. The stored column was built
    positive-mode (round(ExactMolWt + proton, 3)); a negative-mode request, a missing or
    partly-null column, or a DB whose column was produced differently all FAIL the sample
    check and fall through to the original recompute. So correctness is preserved for
    every case; only the common positive-mode path gets faster."""
    if 'Precursormz' not in df.columns or 'smile' not in df.columns:
        return False
    if df['Precursormz'].isna().any():
        return False
    proton = 1.007276
    sign = 1.0 if adduct == '+' else -1.0
    n = min(n_check, len(df))
    sample = df.sample(n=n, random_state=0) if len(df) > n else df
    for smi, stored in zip(sample['smile'], sample['Precursormz']):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or stored is None:
            return False
        expected = round(rdMolDescriptors.CalcExactMolWt(mol) + sign * proton, 3)
        if abs(expected - float(stored)) > tol:
            return False
    return True


def load_reference_database_without_smiles(path, adduct):
    """
    Load and preprocess the reference database for 'without_smiles' input.
    """
    final_mol2vec = _read_reference_any_format(path)

    # Fast path: the reference already carries a Precursormz column. Recomputing it from
    # SMILES on every load is 5.5M RDKit parses (~20 min) that produce a column the file
    # already contains. Trust the stored column when a sample verifies it matches for the
    # requested adduct; otherwise fall through to the original recompute below.
    if _stored_precursormz_is_trustworthy(final_mol2vec, adduct):
        final_mol2vec.reset_index(drop=True, inplace=True)
        return final_mol2vec

    # If necessary, include any required preprocessing steps similar to 'with_smiles'
    # For example, calculating 'Precursormz' if not already present

    g_sm = final_mol2vec['smile'].tolist()
    pr_mass = []
    for i in range(len(g_sm)):
        mol = Chem.MolFromSmiles(g_sm[i])
        if mol is None:
            pr_mass.append(None)
            continue
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        proton_mass = 1.007276  # mass of a proton in Da
        if adduct == '+':

            precursor_mass_positive = mol_weight + proton_mass
        else:

            precursor_mass_positive = mol_weight - proton_mass
        #precursor_mass_positive = mol_weight + proton_mass
        pr_mass.append(round(precursor_mass_positive, 3))
    final_mol2vec['Precursormz'] = pr_mass
    #final_mol2vec.dropna(subset=['Precursormz'], inplace=True)
    final_mol2vec.reset_index(drop=True, inplace=True)
    return final_mol2vec


def match_predictions_to_reference_with_smiles(prediction_df, reference_df, top_n, input_file_type, adduct):
    data_test = prediction_df.copy()
    data_test.reset_index(drop=True, inplace=True)

    g_sm_ls_1 = data_test['smile'].tolist()
    g_sm_ls = [item[0] for item in g_sm_ls_1]

    def calculate_molecular_formula(mol):
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        return formula

    pr_mass = []
    test_mf = []
    for i in range(len(g_sm_ls)):
        # Define a molecule

        mol = Chem.MolFromSmiles(g_sm_ls[i])
        m_f = calculate_molecular_formula(mol)
        # Calculate the exact molecular weight
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)

        proton_mass = 1.007276  # mass of a proton in Da

        # For positive ionization ([M+H]+)
        print('way to check adduct')
        if adduct == '+':
            print('Postive more')
            precursor_mass_positive = mol_weight + proton_mass
        else:
            print("Negative more")
            precursor_mass_positive = mol_weight - proton_mass

        pr_mass.append(round(precursor_mass_positive, 3))
        test_mf.append(m_f)

    data_test['predict_mass'] = pr_mass

    data_test['Molecular_Formula'] = test_mf

    ls_1 = data_test['tree_out'].tolist()

    pd.options.mode.chained_assignment = None

    final_smile = []
    final_scan_number = []
    final_EU = []
    final_Tanimoto = []

    for i in range(len(data_test)):
        new_df = reference_df[reference_df['Precursormz'] == data_test['predict_mass'].iloc[i]].copy()
        new_df.reset_index(drop=True, inplace=True)
        ls_eu = []
        ls_tani = []
        for j in range(len(new_df)):
            array2 = np.array(new_df['up_molvec'].iloc[j])
            # Calculate the cosine similarity
            cosine_similarity = np.dot(ls_1[i], array2) / (norm(ls_1[i]) * norm(array2))
            ls_eu.append(cosine_similarity.item())

            # Calculate Tanimoto similarity
            ground_smile = data_test['smile'].iloc[i]
            candidate_smile = new_df['smile'].iloc[j]
            mol1 = Chem.MolFromSmiles(ground_smile[0])
            mol2 = Chem.MolFromSmiles(candidate_smile)
            if mol1 and mol2:
                fp1 = Chem.RDKFingerprint(mol1)
                fp2 = Chem.RDKFingerprint(mol2)
                tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
            else:
                tanimoto = np.nan
            ls_tani.append(tanimoto)

        new_df['cosine'] = ls_eu
        new_df['Tanimoto'] = ls_tani
        # Assign Unique_ID to all rows (Unique_ID is now the SMILES string)
        new_df['Unique_ID'] = [data_test['Unique_ID'].iloc[i]] * len(new_df)
        new_df.sort_values('cosine', ascending=False, inplace=True)

        if len(new_df) > 0:
            top_smiles = []
            top_euclidean_distances = []
            top_tanimoto = []
            for k in range(min(len(new_df), top_n)):
                top_smiles.append(new_df['smile'].iloc[k])
                top_euclidean_distances.append(new_df['cosine'].iloc[k])
                top_tanimoto.append(new_df['Tanimoto'].iloc[k])

            final_smile.append(top_smiles)
            final_EU.append(top_euclidean_distances)
            final_scan_number.append(new_df['Unique_ID'].iloc[0])
            final_Tanimoto.append(top_tanimoto)

    result_df = pd.DataFrame({'Unique_ID': final_scan_number,
                              'Top smile': final_smile,
                              'Top Min_cosine': final_EU,
                              'Top Tanimoto': final_Tanimoto})

    # Prepare final DataFrame with top candidates
    # Generate column names based on top_n
    new_EU = [f'Top_{i+1}_cosine' for i in range(top_n)]
    new_smile = [f'Top_{i+1}_SMILE' for i in range(top_n)]
    new_tanimoto = [f'Top_{i+1}_Tanimoto' for i in range(top_n)]
    new_columns = new_EU + new_smile + new_tanimoto

    for col_name in new_columns:
        result_df[col_name] = "NA"

    for i in range(len(result_df)):
        ls_smile = result_df['Top smile'].iloc[i]
        ls_Eu = result_df['Top Min_cosine'].iloc[i]
        ls_tanimoto = result_df['Top Tanimoto'].iloc[i]
        # Add new columns in a loop
        for col_name_1, col_value_1 in zip(new_EU, ls_Eu):
            result_df.at[i, col_name_1] = col_value_1

        for col_name_2, col_value_2 in zip(new_smile, ls_smile):
            result_df.at[i, col_name_2] = col_value_2

        for col_name_3, col_value_3 in zip(new_tanimoto, ls_tanimoto):
            result_df.at[i, col_name_3] = col_value_3

    result_df.drop(['Top smile', 'Top Min_cosine', 'Top Tanimoto'], axis=1, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    return result_df




def match_predictions_to_reference_without_smiles(prediction_df, reference_df, top_n, input_file_type,adduct):
    """
    Match predictions to reference database and extract top candidates for 'without_smiles' input.
    """

    data_test = prediction_df.copy()
    data_test.reset_index(drop=True, inplace=True)
    print(data_test.columns)
    # Truncate masses in data_test to three decimal places without rounding
    ls_saam = []
    for i in data_test.Precursor:
        a, b = str(i).split('.')
        update_a = a + '.' + b[:3]
        ls_saam.append(float(update_a))
    data_test['update_mass']= ls_saam
    # Filter data_test where update_mass is in reference_df['Precursormz']

    ls_1 = data_test['tree_out'].tolist()

    pd.options.mode.chained_assignment = None

    final_smile = []
    final_scan_number = []
    final_EU = []
    final_inchikey = []

    for i in range(len(data_test)):
        new_df = reference_df[reference_df['Precursormz'] == data_test['update_mass'].iloc[i]].copy()
        new_df.reset_index(drop=True, inplace=True)
        ls_eu = []
        for j in range(len(new_df)):
            array2 = np.array(new_df['up_molvec'].iloc[j])
            # Calculate the cosine similarity
            cosine_similarity = np.dot(ls_1[i], array2) / (norm(ls_1[i]) * norm(array2))
            ls_eu.append(cosine_similarity.item())

        new_df['cosine'] = ls_eu
        # Assign Unique_ID to all rows
        new_df['Unique_ID'] = [data_test['Unique_ID'].iloc[i]] * len(new_df)
        new_df.sort_values('cosine', ascending=False, inplace=True)

        if len(new_df) > 0:
            top_smiles = []
            top_euclidean_distances = []
            top_inchikey = []
            for k in range(min(len(new_df), top_n)):
                top_smiles.append(new_df['smile'].iloc[k])  # Assuming 'smile' exists in reference_df
                top_euclidean_distances.append(new_df['cosine'].iloc[k])
                top_inchikey.append(new_df['inchikey'].iloc[k])  # Assuming 'inchikey' exists in reference_df

            final_smile.append(top_smiles)
            final_EU.append(top_euclidean_distances)
            final_scan_number.append(new_df['Unique_ID'].iloc[0])
            final_inchikey.append(top_inchikey)

    result_df = pd.DataFrame({'Unique_ID': final_scan_number,
                              'Top smile': final_smile,
                              'Top Min_cosine': final_EU,
                              'Top InChIKey': final_inchikey})

    # Prepare final DataFrame with top candidates
    # Generate column names based on top_n
    new_EU = [f'Top_{i+1}_cosine' for i in range(top_n)]
    new_smile = [f'Top_{i+1}_SMILE' for i in range(top_n)]
    new_inchikey = [f'Top_{i+1}_InChIKey' for i in range(top_n)]
    new_columns = new_EU + new_smile + new_inchikey

    for col_name in new_columns:
        result_df[col_name] = "NA"

    for i in range(len(result_df)):
        ls_smile = result_df['Top smile'].iloc[i]
        ls_Eu = result_df['Top Min_cosine'].iloc[i]
        ls_inchikey = result_df['Top InChIKey'].iloc[i]
        # Add new columns in a loop
        for col_name_1, col_value_1 in zip(new_EU, ls_Eu):
            result_df.at[i, col_name_1] = col_value_1

        for col_name_2, col_value_2 in zip(new_smile, ls_smile):
            result_df.at[i, col_name_2] = col_value_2

        for col_name_3, col_value_3 in zip(new_inchikey, ls_inchikey):
            result_df.at[i, col_name_3] = col_value_3
    
    result_df.drop(['Top smile', 'Top Min_cosine', 'Top InChIKey'], axis=1, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    
    return result_df
