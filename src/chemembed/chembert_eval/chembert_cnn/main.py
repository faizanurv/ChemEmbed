import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from .dataset import class_ls
from .model_cnn import CNN_Class
from numpy.linalg import norm
# Import functions from our modules
from ..data_preprocessing import (
    msp_to_dataframe_with_smiles,
    preprocess_spectra_with_smiles,
    process_data_with_smiles
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained CNN model (.zip or .pth)')
    parser.add_argument('--mspfile', required=True, help='Path to msp file')
    parser.add_argument('--mol2vec', required=True, help='Path to final mol2vec pickle file')
    args = parser.parse_args()

    # === Part 1: Inference ===
    msp_df = msp_to_dataframe_with_smiles(args.mspfile)
    # Parameters
    norm_df = preprocess_spectra_with_smiles(msp_df, 1)

    final_up = process_data_with_smiles(norm_df, 0.01, 0.01, 700)

    test_dataset = class_ls(final_up)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)

    model_cnn = CNN_Class()
    model_cnn.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    with torch.no_grad():
        model_cnn.eval()
        df = pd.DataFrame(columns=('Ground smile', 'tree_out','len_frag','inchikey'))
        count = 0
        for inputs in test_loader:
            inputs01 = inputs[0]
            label1 = inputs[2]
            coo = inputs[3]
            ik = inputs[4]
            outputs1 = model_cnn(inputs01)
            tree_out = outputs1.detach().cpu().numpy()
            coo = coo.detach().numpy().item()
            df.loc[count] = [label1[0], tree_out, coo, ik[0]]
            count += 1
        df.to_pickle('chemberta_final_results_negative.pkl')

    # === Part 2: Ranking & Top-K Calculation ===
    final_mol2vec = pd.read_pickle(args.mol2vec)
    g_sm = final_mol2vec['smiles'].tolist()
    pr_mass = []
    mol_form_mv = []

    def calculate_molecular_formula(mol):
        # mol = Chem.MolFromSmiles(smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        return formula

    for sm in g_sm:
        mol = Chem.MolFromSmiles(sm)
        m_f = calculate_molecular_formula(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        proton_mass = 1.007276
        precursor_mass_positive = mol_weight + proton_mass
        pr_mass.append(round(precursor_mass_positive, 3))
        mol_form_mv.append(m_f)
    final_mol2vec['Precursormz'] = pr_mass
    final_mol2vec['Molecular_Formula'] = mol_form_mv

    data_test = pd.read_pickle('chemberta_final_results_negative.pkl')
    data_test.reset_index(drop=True, inplace=True)
    g_sm_ls = data_test['Ground smile'].tolist()
    pr_mass = []
    test_mf = []
    for sm in g_sm_ls:
        mol = Chem.MolFromSmiles(sm)
        m_f = calculate_molecular_formula(mol)
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        proton_mass = 1.007276
        precursor_mass_positive = mol_weight + proton_mass
        pr_mass.append(round(precursor_mass_positive, 3))
        test_mf.append(m_f)
    data_test['predict_mass'] = pr_mass
    data_test['Molecular_Formula'] = test_mf

    ls_1 = data_test['tree_out'].tolist()
    pd.options.mode.chained_assignment = None

    top_counts = {k: 0 for k in range(1, 12)}
    counts_list = {k: [] for k in range(1, 12)}
    ls_pre_eu = {k: [] for k in range(1, 12)}
    ls_g_sm = {k: [] for k in range(1, 12)}
    ls_p_tani = {k: [] for k in range(1, 12)}
    ls_inchikey = []
    ls_premz = []

    for i in range(len(data_test)):
        new_df = final_mol2vec[final_mol2vec['Precursormz'] == data_test['predict_mass'].iloc[i]]
        new_df.reset_index(drop=True, inplace=True)
        ls_eu = []
        ls_tani = []
        ls_pre_sm = []
        for j in range(len(new_df)):
            array2 = np.array(new_df['cls_embedding'].iloc[j])
            euclidean_distance = np.dot(ls_1[i], array2) / (norm(ls_1[i]) * norm(array2))
            ls_eu.append(euclidean_distance.item())
            mol = Chem.MolFromSmiles(data_test['Ground smile'].iloc[i])
            mol_1 = Chem.MolFromSmiles(new_df['smiles'].iloc[j])
            fp1 = Chem.RDKFingerprint(mol)
            fp2 = Chem.RDKFingerprint(mol_1)
            score = DataStructs.TanimotoSimilarity(fp1, fp2)
            ls_tani.append(score)
            ls_pre_sm.append(data_test['Ground smile'].iloc[i])

        new_df['cosine'] = ls_eu
        new_df['Tanimoto'] = ls_tani
        new_df['pre_smile'] = ls_pre_sm
        new_df.sort_values('cosine', ascending=False, inplace=True)

        placed = False
        for rank in range(1, 6):
            if len(new_df) >= rank and new_df['Tanimoto'].iloc[rank-1] >= 0.95:
                top_counts[rank] += 1
                ls_g_sm[rank].append(new_df['smiles'].iloc[rank-1])
                ls_p_tani[rank].append(new_df['Tanimoto'].iloc[rank-1])
                ls_pre_eu[rank].append(new_df['cosine'].iloc[rank-1])
                if rank == 1:
                    ls_inchikey.append(new_df['inchikey'].iloc[0])
                    ls_premz.append(new_df['Precursormz'].iloc[0])
                counts_list[rank].append(len(new_df))
                placed = True
                break
        if not placed and len(new_df) > 0:
            top_counts[11] += 1
            ls_g_sm[11].append(new_df['smiles'].iloc[0])
            ls_p_tani[11].append(new_df['Tanimoto'].iloc[0])
            ls_pre_eu[11].append(new_df['cosine'].iloc[0])
            counts_list[11].append(len(new_df))

    counts = [top_counts[k] for k in range(1, 6)]
    result1 = [(c * 100) / len(data_test) for c in counts]
    print("Top-1..Top-5 %:", result1)
    print("Sum Top-1..5 %:", sum(result1))

if __name__ == '__main__':
    main()
