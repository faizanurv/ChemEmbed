# reference_utils.py

import pandas as pd
import numpy as np
from math import nan
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm

PROTON_MASS = 1.007276          # Da

# `Precursormz` is stored as round(ExactMolWt +/- PROTON_MASS, 3), so every reference
# mass already carries up to +/-0.0005 Da of quantisation error before any tolerance is
# applied. At m/z 200 that is half of a 5 ppm window, so the window is widened by this
# much to stop the stored rounding from eating the user's tolerance budget.
REFERENCE_MZ_QUANTISATION = 5e-4

# Default precursor tolerance. This is deliberately non-zero: 0 selects the original
# exact-bucket filter, which misses roughly half of all correct candidates (see
# _candidate_positions). Reproducing pre-1.2.0 results requires opting in to 0.
DEFAULT_PRECURSOR_TOLERANCE_PPM = 5.0


def _truncate_mass(mz):
    """Floor-truncate to 3 decimal places, matching ChemEmbed's original query handling."""
    a, _, b = str(float(mz)).partition('.')
    return float(a + '.' + b[:3])


def build_precursor_index(reference_df):
    """Sort the reference precursor masses once so each query is a binary search.

    The original matcher evaluated `reference_df['Precursormz'] == mass` per spectrum,
    a full scan of every reference row for every query. Sorting once and using
    np.searchsorted turns that into O(log R) per query and is what makes a mass WINDOW
    affordable at all -- a window cannot be expressed as an equality test.

    NaN masses (invalid SMILES in the reference) sort to the end and are therefore never
    returned by a window query, which is the behaviour we want.
    """
    masses = pd.to_numeric(reference_df['Precursormz'], errors='coerce').to_numpy(dtype=np.float64)
    order = np.argsort(masses, kind='stable')
    return {'order': order, 'sorted': masses[order], 'n': len(masses)}


def _candidate_positions(index, query_mz, tol_ppm):
    """Row positions in the reference whose precursor matches `query_mz`.

    Two modes:

    * tol_ppm > 0 -- a real mass window, |ref - query| <= query * tol_ppm / 1e6, widened
      by REFERENCE_MZ_QUANTISATION. This is the correct behaviour.

    * tol_ppm == 0 -- the original exact 3-decimal bucket, preserved so old results can
      be reproduced. It has two independent defects. First, the reference is produced by
      round(mass, 3) while the query is produced by truncation, so the equality fails
      whenever the 4th decimal digit is >= 5 -- half of all compounds, even when the
      query mass is exact. Second, a 3-decimal bucket is a +/-0.0005 Da window, about
      1.2 ppm at m/z 400, which is narrower than the mass error of any real instrument.
    """
    if not np.isfinite(query_mz):
        return np.empty(0, dtype=np.intp)

    if tol_ppm and tol_ppm > 0:
        da = abs(query_mz) * float(tol_ppm) / 1e6 + REFERENCE_MZ_QUANTISATION
        lo = np.searchsorted(index['sorted'], query_mz - da, side='left')
        hi = np.searchsorted(index['sorted'], query_mz + da, side='right')
    else:
        # Equality against the stored 3-dp value, expressed as a degenerate range so the
        # sorted index can serve both modes. Selects exactly the same rows as the
        # original boolean mask, only faster.
        bucket = _truncate_mass(query_mz)
        lo = np.searchsorted(index['sorted'], bucket, side='left')
        hi = np.searchsorted(index['sorted'], bucket, side='right')

    return index['order'][lo:hi]


def _cosine_against(candidate_vectors, query_vector):
    """Cosine similarity of one query against a stack of candidate embeddings."""
    if len(candidate_vectors) == 0:
        return np.empty(0, dtype=np.float64)
    mat = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in candidate_vectors])
    q = np.asarray(query_vector, dtype=np.float64).ravel()
    denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, mat @ q / denom, np.nan)


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


def match_predictions_to_reference_with_smiles(prediction_df, reference_df, top_n,
                                               input_file_type, adduct,
                                               tol_ppm=DEFAULT_PRECURSOR_TOLERANCE_PPM):
    """Match predictions for 'with_smiles' input, scoring cosine and Tanimoto.

    Note that this path derives the query mass from the known SMILES with the same
    round(.,3) the reference uses, so it never suffered the round-versus-truncate
    mismatch that afflicts the 'without_smiles' path. It still benefits from a real
    tolerance window, and from no longer dropping unmatched spectra.
    """
    data_test = prediction_df.reset_index(drop=True)

    query_mass = []
    formulas = []
    sign = 1.0 if adduct == '+' else -1.0
    for value in data_test['smile'].tolist():
        # 'smile' arrives as a one-element sequence from the DataLoader.
        smi = value[0] if isinstance(value, (list, tuple)) else value
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            # Previously CalcMolFormula was called on None, so a single unparseable
            # SMILES aborted the whole run.
            query_mass.append(np.nan)
            formulas.append(None)
            continue
        query_mass.append(rdMolDescriptors.CalcExactMolWt(mol) + sign * PROTON_MASS)
        formulas.append(rdMolDescriptors.CalcMolFormula(mol))

    query_mass = np.asarray(query_mass, dtype=np.float64)
    data_test['predict_mass'] = np.round(query_mass, 3)
    data_test['Molecular_Formula'] = formulas

    index = build_precursor_index(reference_df)
    ref_smile = reference_df['smile'].to_numpy()
    ref_vecs = reference_df['up_molvec'].to_numpy()

    predictions = data_test['tree_out'].tolist()

    final_uid, final_smile, final_cosine, final_tanimoto = [], [], [], []
    n_unmatched = 0

    for i in range(len(data_test)):
        rows = _candidate_positions(index, query_mass[i], tol_ppm)

        if len(rows) == 0:
            n_unmatched += 1
            final_uid.append(data_test['Unique_ID'].iloc[i])
            final_cosine.append([])
            final_smile.append([])
            final_tanimoto.append([])
            continue

        cosines = _cosine_against(ref_vecs[rows], predictions[i])
        take = min(len(rows), top_n)
        best = np.argsort(-cosines, kind='stable')[:take]
        sel = rows[best]

        # Tanimoto only for the candidates actually reported, rather than for every row
        # in the bucket: the fingerprints are the expensive part and most are discarded.
        truth = data_test['smile'].iloc[i]
        truth_smi = truth[0] if isinstance(truth, (list, tuple)) else truth
        mol1 = Chem.MolFromSmiles(truth_smi) if isinstance(truth_smi, str) else None
        fp1 = Chem.RDKFingerprint(mol1) if mol1 is not None else None

        tanimotos = []
        for j in sel:
            mol2 = Chem.MolFromSmiles(ref_smile[j]) if isinstance(ref_smile[j], str) else None
            if fp1 is not None and mol2 is not None:
                tanimotos.append(DataStructs.TanimotoSimilarity(fp1, Chem.RDKFingerprint(mol2)))
            else:
                tanimotos.append(np.nan)

        final_uid.append(data_test['Unique_ID'].iloc[i])
        final_cosine.append(cosines[best].tolist())
        final_smile.append(list(ref_smile[sel]))
        final_tanimoto.append(tanimotos)

    mode = f"+/-{tol_ppm:g} ppm window" if tol_ppm and tol_ppm > 0 else "exact 3-dp bucket (legacy)"
    print(f"[match] precursor filter: {mode}")
    print(f"[match] {len(data_test)} spectra processed, "
          f"{len(data_test) - n_unmatched} matched, "
          f"{n_unmatched} with no reference within the precursor tolerance")

    result_df = pd.DataFrame({'Unique_ID': final_uid,
                              'Top smile': final_smile,
                              'Top Min_cosine': final_cosine,
                              'Top Tanimoto': final_tanimoto})

    return _widen_to_top_n_columns(result_df, top_n, {
        'Top Min_cosine': ('cosine', nan),
        'Top smile': ('SMILE', "NA"),
        'Top Tanimoto': ('Tanimoto', nan),
    })




def _widen_to_top_n_columns(result_df, top_n, list_columns):
    """Explode per-row candidate lists into fixed Top_1..Top_N columns.

    `list_columns` maps the temporary list column to (prefix, fill). Cosine fills with
    NaN so the column stays a float dtype and remains sortable; identifier columns fill
    with "NA".
    """
    to_drop = []
    for src, (prefix, fill) in list_columns.items():
        names = [f'Top_{i+1}_{prefix}' for i in range(top_n)]
        for name in names:
            result_df[name] = fill
        for i in range(len(result_df)):
            for name, value in zip(names, result_df[src].iloc[i]):
                result_df.at[i, name] = value
        to_drop.append(src)

    result_df.drop(to_drop, axis=1, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def match_predictions_to_reference_without_smiles(prediction_df, reference_df, top_n,
                                                  input_file_type, adduct,
                                                  tol_ppm=DEFAULT_PRECURSOR_TOLERANCE_PPM):
    """
    Match predictions to reference database and extract top candidates for 'without_smiles' input.

    Every input spectrum produces exactly one output row. A spectrum with no reference
    within the precursor tolerance comes back with "NA" candidates and NaN cosines
    rather than being dropped: previously such spectra vanished from the CSV silently,
    so a run over 1000 spectra could return 300 rows with no error and no indication of
    what had happened to the rest.
    """
    data_test = prediction_df.reset_index(drop=True)

    index = build_precursor_index(reference_df)
    ref_smile = reference_df['smile'].to_numpy() if 'smile' in reference_df.columns else None
    ref_ik = reference_df['inchikey'].to_numpy() if 'inchikey' in reference_df.columns else None
    ref_vecs = reference_df['up_molvec'].to_numpy()

    # The window path compares real masses, so the query must NOT be floored -- flooring
    # is half of the defect the window exists to fix. The legacy bucket path still needs
    # the truncated form.
    query_mass = pd.to_numeric(data_test['Precursor'], errors='coerce').to_numpy(dtype=np.float64)

    predictions = data_test['tree_out'].tolist()

    final_uid, final_smile, final_cosine, final_inchikey = [], [], [], []
    n_unmatched = 0

    for i in range(len(data_test)):
        rows = _candidate_positions(index, query_mass[i], tol_ppm)

        if len(rows) == 0:
            n_unmatched += 1
            final_uid.append(data_test['Unique_ID'].iloc[i])
            final_cosine.append([])
            final_smile.append([])
            final_inchikey.append([])
            continue

        cosines = _cosine_against(ref_vecs[rows], predictions[i])
        take = min(len(rows), top_n)
        best = np.argsort(-cosines, kind='stable')[:take]
        sel = rows[best]

        final_uid.append(data_test['Unique_ID'].iloc[i])
        final_cosine.append(cosines[best].tolist())
        final_smile.append(list(ref_smile[sel]) if ref_smile is not None else ["NA"] * take)
        final_inchikey.append(list(ref_ik[sel]) if ref_ik is not None else ["NA"] * take)

    mode = f"+/-{tol_ppm:g} ppm window" if tol_ppm and tol_ppm > 0 else "exact 3-dp bucket (legacy)"
    print(f"[match] precursor filter: {mode}")
    print(f"[match] {len(data_test)} spectra processed, "
          f"{len(data_test) - n_unmatched} matched, "
          f"{n_unmatched} with no reference within the precursor tolerance")

    result_df = pd.DataFrame({'Unique_ID': final_uid,
                              'Top smile': final_smile,
                              'Top Min_cosine': final_cosine,
                              'Top InChIKey': final_inchikey})

    return _widen_to_top_n_columns(result_df, top_n, {
        'Top Min_cosine': ('cosine', nan),
        'Top smile': ('SMILE', "NA"),
        'Top InChIKey': ('InChIKey', "NA"),
    })
