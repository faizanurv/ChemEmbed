# data_processing.py

import pandas as pd
import numpy as np
from rdkit import Chem
import scipy.sparse

def msp_to_dataframe_with_smiles(msp_file):
    """
    Convert an MSP file with SMILES to a pandas DataFrame.
    """
    data = {"smile": [], "Precursor": [], "Adduct": [], "spectra": []}
    with open(msp_file, 'r') as f:
        current_smile = None
        current_precursor = None
        current_adduct = None
        current_spectra = []

        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith("smile:"):
                if current_smile:
                    # Save previous spectrum
                    data["smile"].append(current_smile)
                    data["Precursor"].append(current_precursor)
                    data["Adduct"].append(current_adduct)
                    data["spectra"].append(current_spectra)
                # Start new spectrum
                current_smile = line.split(":", 1)[1].strip()
                current_spectra = []
            elif line.startswith("Precursor:"):
                current_precursor = float(line.split(":", 1)[1].strip())
            elif line.startswith("Adduct:"):
                current_adduct = line.split(":", 1)[1].strip()
            elif line[0].isdigit():
                mz_intensity = line.split()
                if len(mz_intensity) >= 2:
                    mz = float(mz_intensity[0])
                    intensity = float(mz_intensity[1])
                    current_spectra.append((mz, intensity))
        # Append the last spectrum
        if current_smile:
            data["smile"].append(current_smile)
            data["Precursor"].append(current_precursor)
            data["Adduct"].append(current_adduct)
            data["spectra"].append(current_spectra)

    df = pd.DataFrame(data)
    return df

def msp_to_dataframe_without_smiles(msp_file):
    """
    Convert an MSP file without SMILES to a pandas DataFrame.
    """
    data = {"Unique_ID": [], "Precursor": [], "Adduct": [], "spectra": []}
    with open(msp_file, 'r') as f:
        current_id = None
        current_precursor = None
        current_adduct = None
        current_spectra = []

        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith("Name:"):
                if current_id:
                    # Save previous spectrum
                    data["Unique_ID"].append(current_id)
                    data["Precursor"].append(current_precursor)
                    data["Adduct"].append(current_adduct)
                    data["spectra"].append(current_spectra)
                # Start new spectrum
                current_id = line.split(":", 1)[1].strip()
                current_spectra = []
            elif line.startswith("Precursor:"):
                current_precursor = float(line.split(":", 1)[1].strip())
            elif line.startswith("Adduct:"):
                current_adduct = line.split(":", 1)[1].strip()
            elif line[0].isdigit():
                mz_intensity = line.split()
                if len(mz_intensity) >= 2:
                    mz = float(mz_intensity[0])
                    intensity = float(mz_intensity[1])
                    current_spectra.append((mz, intensity))
        # Append the last spectrum
        if current_id:
            data["Unique_ID"].append(current_id)
            data["Precursor"].append(current_precursor)
            data["Adduct"].append(current_adduct)
            data["spectra"].append(current_spectra)

    df = pd.DataFrame(data)
    return df

def normalized_intensity(intensity, max_value):
    temp = intensity / max_value
    ni = temp * 100
    return ni

def step_three_preprocessing(value, max_value):
    temp = value / max_value
    ni_r = temp * 100
    return ni_r

def normalize_by_sum(value, sumv):
    n_v = value / sumv
    return n_v

def preprocess_spectra_with_smiles(df_spec, intensity_threshold):
    """
    Preprocess the spectra data for 'with_smiles' input.
    """
    ls_temp1 = []
    ls_smile = []
    ls_mass = []

    for i in range(len(df_spec)):
        ls_temp = []
        for j in range(len(df_spec["spectra"].iloc[i])):
            mass = float(df_spec["Precursor"].iloc[i])
            mz = float(df_spec["spectra"].iloc[i][j][0])
            intensity = float(df_spec["spectra"].iloc[i][j][1])
            final_mz = (mass + 0.5)
            if final_mz >= mz:
                list_spec = np.array([mz, intensity])
                ls_temp.append(list_spec)
        if ls_temp != []:
            ls_mass.append(df_spec["Precursor"].iloc[i])
            ls_smile.append(df_spec["smile"].iloc[i])
            ls_temp1.append(ls_temp)

    ls_int = []
    for i in range(len(ls_temp1)):
        ls_int1 = []
        for j in range(len(ls_temp1[i])):
            intensity = ls_temp1[i][j][1]
            ls_int1.append(intensity)
        ls_int.append(ls_int1)

    for k in range(len(ls_temp1)):
        max_value = max(ls_int[k])
        for ll in range(len(ls_int[k])):
            intensit = ls_int[k][ll]
            ni = normalized_intensity(intensit, max_value)
            ls_int[k][ll] = ni

    step_three_lst = []
    for k in range(len(ls_temp1)):
        sum_n1 = max(ls_int[k])
        step_three_lst1 = []
        for ll in range(len(ls_int[k])):
            value = ls_int[k][ll]
            ni_r = step_three_preprocessing(value, sum_n1)
            if ni_r >= intensity_threshold:
                mass = ls_temp1[k][ll][0]
                arr = np.array([mass, value])
                step_three_lst1.append(arr)
        step_three_lst.append(step_three_lst1)

    spec_lst = []
    for i in range(len(step_three_lst)):
        spec_lst1 = []
        for j in range(len(step_three_lst[i])):
            spec_lst1.append(step_three_lst[i][j][1])
        spec_lst.append(spec_lst1)

    for k in range(len(step_three_lst)):
        sumv = sum(spec_lst[k])
        for ll in range(len(step_three_lst[k])):
            n_v = normalize_by_sum(step_three_lst[k][ll][1], sumv)
            step_three_lst[k][ll][1] = n_v

    norm_df = pd.DataFrame({
        'smile': ls_smile,
        'Precursor': ls_mass,
        "spectra": step_three_lst
    })

    return norm_df

def preprocess_spectra_without_smiles(df_spec, intensity_threshold):
    """
    Preprocess the spectra data for 'without_smiles' input.
    """
    ls_temp1 = []
    ls_mass = []
    ls_unique_id = []

    for i in range(len(df_spec)):
        ls_temp = []
        for j in range(len(df_spec["spectra"].iloc[i])):
            mass = float(df_spec["Precursor"].iloc[i])
            mz = float(df_spec["spectra"].iloc[i][j][0])
            intensity = float(df_spec["spectra"].iloc[i][j][1])
            final_mz = (mass + 0.5)
            if final_mz >= mz:
                list_spec = np.array([mz, intensity])
                ls_temp.append(list_spec)
        if ls_temp != []:
            ls_mass.append(df_spec["Precursor"].iloc[i])
            ls_unique_id.append(df_spec["Unique_ID"].iloc[i])
            ls_temp1.append(ls_temp)

    ls_int = []
    for i in range(len(ls_temp1)):
        ls_int1 = []
        for j in range(len(ls_temp1[i])):
            intensity = ls_temp1[i][j][1]
            ls_int1.append(intensity)
        ls_int.append(ls_int1)

    for k in range(len(ls_temp1)):
        max_value = max(ls_int[k])
        for ll in range(len(ls_int[k])):
            intensit = ls_int[k][ll]
            ni = normalized_intensity(intensit, max_value)
            ls_int[k][ll] = ni

    step_three_lst = []
    for k in range(len(ls_temp1)):
        sum_n1 = max(ls_int[k])
        step_three_lst1 = []
        for ll in range(len(ls_int[k])):
            value = ls_int[k][ll]
            ni_r = step_three_preprocessing(value, sum_n1)
            if ni_r >= intensity_threshold:
                mass = ls_temp1[k][ll][0]
                arr = np.array([mass, value])
                step_three_lst1.append(arr)
        step_three_lst.append(step_three_lst1)

    spec_lst = []
    for i in range(len(step_three_lst)):
        spec_lst1 = []
        for j in range(len(step_three_lst[i])):
            spec_lst1.append(step_three_lst[i][j][1])
        spec_lst.append(spec_lst1)

    for k in range(len(step_three_lst)):
        sumv = sum(spec_lst[k])
        for ll in range(len(step_three_lst[k])):
            n_v = normalize_by_sum(step_three_lst[k][ll][1], sumv)
            step_three_lst[k][ll][1] = n_v

    norm_df = pd.DataFrame({
        'Unique_ID': ls_unique_id,
        'Precursor': ls_mass,
        "spectra": step_three_lst
    })

    return norm_df

def group_measurements_and_sum_intensities(measurements, intensities, tolerance=0.01):
    """
    Group measurements within a tolerance and sum their intensities.
    """
    combined_data = sorted(zip(measurements, intensities), key=lambda x: x[0])
    grouped_data = []
    current_group = [combined_data[0]]

    for i in range(1, len(combined_data)):
        if abs(combined_data[i][0] - combined_data[i - 1][0]) <= tolerance:
            current_group.append(combined_data[i])
        else:
            mz_mean = sum(mz for mz, _ in current_group) / len(current_group)
            intensity_sum = sum(intensity for _, intensity in current_group)
            grouped_data.append((round(mz_mean, 3), intensity_sum))
            current_group = [combined_data[i]]

    if current_group:
        mz_mean = sum(mz for mz, _ in current_group) / len(current_group)
        intensity_sum = sum(intensity for _, intensity in current_group)
        grouped_data.append((round(mz_mean, 3), intensity_sum))

    return grouped_data

def align_spectra(resolution, max_mz, spectrum):
    """
    Align spectra to a fixed resolution.
    """
    numbers = np.arange(0, max_mz, step=resolution)
    result = [0] * len(numbers)
    for i in spectrum:
        idx = np.searchsorted(numbers, i[0])
        if idx >= len(result):
            idx = len(result) - 1
        result[idx] = 1  # i[1]
    return np.array(result)

def calculate_neutral_loss(parent_mz, daughter_mz):
    """
    Calculate the neutral loss between parent and daughter ions.
    """
    return parent_mz - daughter_mz

def process_data_with_smiles(norm_df, tolerance, resolution, max_mz):
    """
    Process normalized data to prepare for model input for 'with_smiles' input.
    """
    # Additional processing steps for 'with_smiles'
    # norm_df['up_inchikey'] = norm_df['smile'].apply(lambda x: Chem.InchiToInchiKey(Chem.MolToInchi(Chem.MolFromSmiles(x)))[:14])

    # Drop duplicates based on 'up_inchikey'
    # unique_keys_df = norm_df[['up_inchikey', 'smile', 'Precursor']].drop_duplicates('up_inchikey')
    # Check if value is NaN using pd.isna()
    norm_df['up_inchikey'] = "Nan"
    for i in range(len(norm_df)):
        mol = Chem.MolFromSmiles(norm_df['smile'].iloc[i])
        # Generate the InChIKey
        inchikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol))

        norm_df.loc[i, 'up_inchikey'] = inchikey[:14]

    list_data = [norm_df]
    # Assuming `list_data` is a list of DataFrames with the columns: ['up_inchikey', 'spectra', 'smile', 'Precursor']
    list_final_data = []

    for data in list_data:
        # Drop duplicates based on 'up_inchikey' to work with unique keys directly
        unique_keys_df = data[['up_inchikey', 'smile', 'Precursor']].drop_duplicates('up_inchikey')

        # Aggregate spectra and smiles by 'up_inchikey'
        aggregated = data.groupby('up_inchikey').agg({
            'spectra': lambda x: [y for sublist in x for y in sublist],  # Flatten the list of lists
            'smile': 'first'  # Assuming the first smile per up_inchikey is representative
        }).reset_index()

        # Merge with unique_keys_df to get 'Precursor' and 'smile' columns aligned with aggregated data
        final_df = pd.merge(aggregated, unique_keys_df, on='up_inchikey', suffixes=('', '_drop')).drop(['smile_drop'],
                                                                                                       axis=1)

        # At this point, 'final_df' contains unique 'up_inchikey' rows with aggregated 'spectra', and first 'smile' and 'Precursor' values.

        # Split the spectra into 'mz' and 'ints'
        final_df['mz'], final_df['ints'] = zip(*final_df['spectra'].apply(lambda x: zip(*x) if x else ([], [])))

        # Dropping 'spectra' as it's already split into 'mz' and 'ints'
        final_df.drop(['spectra'], axis=1, inplace=True)

        # Drop duplicates based on 'smile'
        final_df.drop_duplicates(subset=['smile'], inplace=True)

        list_final_data.append(final_df)
    final_df_pr_spec = list_final_data[0]
    '''aggregated = norm_df.groupby('smile').agg({
        'spectra': lambda x: [y for sublist in x for y in sublist],
        'Precursor': 'first'
    }).reset_index()'''

    # final_df = pd.merge(aggregated, unique_keys_df, on='up_inchikey', suffixes=('', '_drop')).drop(['smile_drop'], axis=1)

    # Split the spectra into 'mz' and 'ints'
    #aggregated['mz'], aggregated['ints'] = zip(*aggregated['spectra'].apply(lambda x: list(zip(*x)) if x else ([], [])))
    #aggregated.drop(['spectra'], axis=1, inplace=True)
    #aggregated.drop_duplicates(subset=['smile'], inplace=True)
    final_df_pr_spec.reset_index(drop=True, inplace=True)

    # Continue processing similar to 'without_smiles'
    return _process_common(final_df_pr_spec, tolerance, resolution, max_mz)

def process_data_without_smiles(norm_df, tolerance, resolution, max_mz):
    """
    Process normalized data to prepare for model input for 'without_smiles' input.
    """
    norm_df['mz'] = norm_df['spectra'].apply(lambda x: [item[0] for item in x])
    norm_df['ints'] = norm_df['spectra'].apply(lambda x: [item[1] for item in x])
    norm_df.drop(['spectra'], axis=1, inplace=True)
    norm_df.drop_duplicates(subset=['Unique_ID'], inplace=True)
    norm_df.reset_index(drop=True, inplace=True)

    # Continue processing similar to 'with_smiles'
    return _process_common(norm_df, tolerance, resolution, max_mz)

def _process_common(df, tolerance, resolution, max_mz):
    """
    Common processing steps for both input types.
    """
    f_up_mz = []
    f_up_ints = []
    for i in range(len(df)):
        grouped_data = group_measurements_and_sum_intensities(df['mz'].iloc[i], df['ints'].iloc[i], tolerance)
        up_mz = []
        up_ints = []
        for group in grouped_data:
            up_mz.append(round(group[0], 2))
            up_ints.append(group[1])
        f_up_mz.append(up_mz)
        f_up_ints.append(up_ints)
    df['up_mz'] = f_up_mz
    df['up_ints'] = f_up_ints

    # Calculate neutral loss
    pr_lst = []
    for i in range(len(df)):
        pr_mz = df['Precursor'].iloc[i]
        pr_mz = round(pr_mz, 3)
        pr_mz = pr_mz + 0.5
        sub_lst = []
        for s_peak in df['up_mz'].iloc[i]:
            s_peak = round(s_peak, 3)
            sub_lst.append(round(calculate_neutral_loss(pr_mz, s_peak), 3))
        pr_lst.append(sub_lst)
    df['neutral_loss'] = pr_lst

    # Combine m/z and neutral loss
    up_lst = []
    for i in range(len(df)):
        up_mz_ls = df['up_mz'].iloc[i]
        up_nl_ls = df['neutral_loss'].iloc[i]
        combined_list = up_mz_ls + up_nl_ls
        up_lst.append(combined_list)
    df['up_mz_nl'] = up_lst

    # Filter data
    final_up = df[df['Precursor'] <= 1000]

    # Align spectra
    main_ls = []
    for i in range(len(final_up)):
        mz1 = final_up['up_mz_nl'].iloc[i]
        ints1 = final_up['up_mz_nl'].iloc[i]
        fs = []
        for j in range(len(mz1)):
            fs.append((mz1[j], ints1[j]))
        main_ls.append(fs)

    new_spec = []
    for i in range(len(main_ls)):
        spec = align_spectra(resolution, max_mz, main_ls[i])
        new_spec.append(spec)

    # Convert to sparse format
    sparse = []
    for spec in new_spec:
        sparse.append(scipy.sparse.csr_matrix(spec))
    final_up['coo_format_data'] = sparse

    final_up.drop(['up_mz', 'up_ints', 'neutral_loss', 'up_mz_nl'], axis=1, inplace=True)
    final_up.reset_index(drop=True, inplace=True)
    return final_up
