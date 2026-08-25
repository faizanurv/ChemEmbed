"""Precursor candidate retrieval.

Every test here corresponds to a measured defect in the 3-decimal exact-bucket filter
that shipped up to 1.1.1, or to a property of the ppm window that replaced it. These
need only numpy/pandas/rdkit -- no torch, no models, no 520k-row reference database --
so they run in seconds and are safe to make a required check.
"""
import numpy as np
import pandas as pd
import pytest

from chemembed.reference_utils import (
    DEFAULT_PRECURSOR_TOLERANCE_PPM,
    PROTON_MASS,
    REFERENCE_MZ_QUANTISATION,
    _candidate_positions,
    _truncate_mass,
    build_precursor_index,
    match_predictions_to_reference_without_smiles,
)

D = 300


def _reference(masses, vectors=None):
    """A minimal reference database. `masses` are THEORETICAL precursor masses; they are
    stored round(.,3) exactly as load_reference_database_without_smiles does."""
    n = len(masses)
    if vectors is None:
        vectors = [np.full(D, float(i + 1)) for i in range(n)]
    return pd.DataFrame({
        "smile": [f"SMILES_{i}" for i in range(n)],
        "inchikey": [f"KEY{i:011d}" for i in range(n)],
        "up_molvec": vectors,
        "Precursormz": [round(m, 3) for m in masses],
    })


def _predictions(rows):
    """rows: list of (unique_id, measured_precursor, embedding)."""
    return pd.DataFrame({
        "Unique_ID": [r[0] for r in rows],
        "Precursor": [r[1] for r in rows],
        "tree_out": [r[2] for r in rows],
    })


# ---------------------------------------------------------------------------
# Defect 1: the reference rounds and the query truncates, so exact equality
# fails whenever the 4th decimal digit is >= 5 -- half of all compounds, even
# when the query mass is perfect. Measured at 49.9% over random masses.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mass", [194.0806, 230.0048, 118.0865, 512.29951])
def test_rounding_and_truncation_disagree_for_high_fourth_decimal(mass):
    assert round(mass, 3) != _truncate_mass(mass)


@pytest.mark.parametrize("mass", [194.0801, 230.0051, 118.0863, 300.1234])
def test_rounding_and_truncation_agree_for_low_fourth_decimal(mass):
    assert round(mass, 3) == _truncate_mass(mass)


def test_legacy_bucket_misses_a_compound_at_its_own_exact_mass():
    """The compound is in the database, the query mass is exact, and the legacy
    filter still returns nothing. This is the headline defect."""
    ref = _reference([230.0048])
    index = build_precursor_index(ref)

    assert len(_candidate_positions(index, 230.0048, tol_ppm=0)) == 0
    assert len(_candidate_positions(index, 230.0048, tol_ppm=5.0)) == 1


def test_window_recovers_every_compound_at_its_exact_mass():
    """Across masses whose 4th decimal spans 0-9, the legacy filter retrieves its OWN
    reference row about half the time and the window retrieves it every time.

    Note this asserts that the compound's own row comes back, not merely that some row
    does. In a dense database a wrong-bucket query still lands on a neighbour, so a
    non-emptiness check would pass while returning the wrong compound entirely.
    """
    # Spaced far enough apart that every compound occupies its own 3-dp bucket, so a
    # retrieved row can only be the right one.
    masses = [100.0 + i * 0.05 + (i % 10) * 0.0001 for i in range(200)]
    ref = _reference(masses)
    index = build_precursor_index(ref)

    legacy = sum(i in _candidate_positions(index, m, 0) for i, m in enumerate(masses))
    window = sum(i in _candidate_positions(index, m, 5.0) for i, m in enumerate(masses))

    assert window == len(masses)
    assert legacy < 0.75 * len(masses)   # measured ~50%, assert well clear of "fine"


# ---------------------------------------------------------------------------
# Defect 2: a 3-decimal bucket is +/-0.0005 Da, about 1.2 ppm at m/z 400, which
# is narrower than the mass accuracy of any real instrument. A realistic
# measurement falls into a neighbouring bucket and matches nothing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ppm_error", [2.0, 3.0, 4.0])
def test_realistic_mass_error_falls_out_of_the_legacy_bucket(ppm_error):
    theoretical = 400.12341                      # 4th decimal < 5, so the legacy
    ref = _reference([theoretical])              # filter would match a PERFECT query
    index = build_precursor_index(ref)

    assert len(_candidate_positions(index, theoretical, 0)) == 1, "sanity: exact query matches"

    measured = theoretical * (1 + ppm_error / 1e6)
    assert len(_candidate_positions(index, measured, 0)) == 0
    assert len(_candidate_positions(index, measured, 5.0)) == 1


# ---------------------------------------------------------------------------
# The window must stay a window: a candidate well outside the tolerance is not
# allowed back in. A fix that simply widened the net would pass the tests above.
# ---------------------------------------------------------------------------
def test_window_excludes_candidates_beyond_the_tolerance():
    theoretical = 400.1234
    ref = _reference([theoretical, theoretical * (1 + 50 / 1e6)])   # 2nd is 50 ppm away
    index = build_precursor_index(ref)

    hits = _candidate_positions(index, theoretical, tol_ppm=5.0)
    assert len(hits) == 1
    assert ref["Precursormz"].to_numpy()[hits][0] == pytest.approx(round(theoretical, 3))


def test_window_widens_with_the_requested_tolerance():
    theoretical = 400.1234
    ref = _reference([theoretical, theoretical * (1 + 30 / 1e6)])
    index = build_precursor_index(ref)

    assert len(_candidate_positions(index, theoretical, 5.0)) == 1
    assert len(_candidate_positions(index, theoretical, 50.0)) == 2


# ---------------------------------------------------------------------------
# The stored Precursormz is round(mass, 3), so it carries up to 0.0005 Da of
# quantisation before any tolerance applies. At m/z 200 that is half of a 5 ppm
# window. The window is widened to compensate.
# ---------------------------------------------------------------------------
def test_stored_rounding_does_not_consume_the_tolerance_at_low_mass():
    """Storage rounds the mass UP, the measurement sits BELOW it, and the two errors
    add. Without the quantisation term the true compound falls outside its own window."""
    theoretical = 150.0006                        # rounds up to 150.001, i.e. +0.0004
    ref = _reference([theoretical])
    index = build_precursor_index(ref)
    assert ref["Precursormz"].iloc[0] == pytest.approx(150.001)

    tol_ppm = 3.0
    measured = theoretical * (1 - tol_ppm / 1e6)  # 3 ppm low: a further -0.00045 Da
    gap = abs(ref["Precursormz"].iloc[0] - measured)

    # A window of tol_ppm alone is not wide enough to bridge storage + measurement...
    assert gap > measured * tol_ppm / 1e6
    # ...but it is once the known storage quantisation is added back.
    assert gap <= measured * tol_ppm / 1e6 + REFERENCE_MZ_QUANTISATION
    assert len(_candidate_positions(index, measured, tol_ppm)) == 1


def test_quantisation_constant_matches_three_decimal_rounding():
    assert REFERENCE_MZ_QUANTISATION == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# The sorted index must select exactly what the original boolean mask selected,
# for the legacy mode -- it is a speed change there, not a behaviour change.
# ---------------------------------------------------------------------------
def test_legacy_mode_reproduces_the_original_boolean_mask():
    masses = [100.0 + i * 0.0007 for i in range(300)]
    ref = _reference(masses)
    index = build_precursor_index(ref)
    stored = ref["Precursormz"].to_numpy()

    for query in masses[::7]:
        bucket = _truncate_mass(query)
        expected = set(np.flatnonzero(stored == bucket).tolist())
        actual = set(_candidate_positions(index, query, tol_ppm=0).tolist())
        assert actual == expected


def test_index_tolerates_unparseable_reference_rows():
    """Invalid SMILES leave Precursormz as None, which must sort out of the way
    rather than crash the search or be returned as a candidate."""
    ref = _reference([200.1234, 300.5678])
    ref.loc[1, "Precursormz"] = None
    index = build_precursor_index(ref)

    assert len(_candidate_positions(index, 200.1234, 5.0)) == 1
    assert len(_candidate_positions(index, 300.5678, 5.0)) == 0


def test_non_finite_query_returns_no_candidates_rather_than_raising():
    index = build_precursor_index(_reference([200.1234]))
    assert len(_candidate_positions(index, float("nan"), 5.0)) == 0


# ---------------------------------------------------------------------------
# Defect 3: spectra with no candidate were dropped from the output entirely --
# no row, no warning. A 1000-spectrum run could return 300 rows and look clean.
# ---------------------------------------------------------------------------
def test_every_input_spectrum_produces_exactly_one_output_row():
    ref = _reference([200.1234, 300.5678])
    preds = _predictions([
        ("SPEC_HIT_1", 200.1234, np.full(D, 1.0)),
        ("SPEC_NO_MATCH", 999.9999, np.full(D, 1.0)),   # nothing within any tolerance
        ("SPEC_HIT_2", 300.5678, np.full(D, 1.0)),
    ])

    out = match_predictions_to_reference_without_smiles(preds, ref, 3, "without_smiles", "+")

    assert len(out) == 3
    assert out["Unique_ID"].tolist() == ["SPEC_HIT_1", "SPEC_NO_MATCH", "SPEC_HIT_2"]


def test_unmatched_spectrum_is_reported_as_na_not_omitted():
    ref = _reference([200.1234])
    preds = _predictions([("SPEC_NO_MATCH", 999.9999, np.full(D, 1.0))])

    out = match_predictions_to_reference_without_smiles(preds, ref, 2, "without_smiles", "+")

    assert len(out) == 1
    assert out["Top_1_SMILE"].iloc[0] == "NA"
    assert out["Top_1_InChIKey"].iloc[0] == "NA"
    assert np.isnan(out["Top_1_cosine"].iloc[0])


def test_unmatched_count_is_reported_to_the_user(capsys):
    """Silence was the real problem: the count has to reach stdout."""
    ref = _reference([200.1234])
    preds = _predictions([
        ("A", 200.1234, np.full(D, 1.0)),
        ("B", 999.9999, np.full(D, 1.0)),
    ])

    match_predictions_to_reference_without_smiles(preds, ref, 1, "without_smiles", "+")

    out = capsys.readouterr().out
    assert "2 spectra processed" in out
    assert "1 matched" in out
    assert "1 with no reference" in out


# ---------------------------------------------------------------------------
# End-to-end: the right compound comes back top-1, ranked by cosine.
# ---------------------------------------------------------------------------
def test_correct_candidate_is_ranked_first_by_cosine():
    target = np.zeros(D); target[0] = 1.0
    decoy_a = np.zeros(D); decoy_a[1] = 1.0
    decoy_b = np.zeros(D); decoy_b[0] = 1.0; decoy_b[1] = 1.0

    mass = 250.0806                       # 4th decimal >= 5: legacy would miss this
    ref = _reference([mass, mass + 1e-6, mass - 1e-6], vectors=[decoy_a, target, decoy_b])

    preds = _predictions([("QUERY", mass, target)])
    out = match_predictions_to_reference_without_smiles(preds, ref, 3, "without_smiles", "+")

    assert out["Top_1_SMILE"].iloc[0] == "SMILES_1"          # the target row
    assert out["Top_1_cosine"].iloc[0] == pytest.approx(1.0)
    assert out["Top_1_cosine"].iloc[0] >= out["Top_2_cosine"].iloc[0]
    assert out["Top_2_cosine"].iloc[0] >= out["Top_3_cosine"].iloc[0]


def test_legacy_mode_still_reachable_for_reproducing_old_results():
    mass = 250.0806
    ref = _reference([mass])
    preds = _predictions([("QUERY", mass, np.full(D, 1.0))])

    legacy = match_predictions_to_reference_without_smiles(
        preds, ref, 1, "without_smiles", "+", tol_ppm=0)
    assert legacy["Top_1_SMILE"].iloc[0] == "NA"             # the old, wrong answer

    fixed = match_predictions_to_reference_without_smiles(
        preds, ref, 1, "without_smiles", "+", tol_ppm=5.0)
    assert fixed["Top_1_SMILE"].iloc[0] == "SMILES_0"


def test_cosine_columns_stay_float_when_candidates_are_missing():
    """Filling identifiers with "NA" must not make the cosine column object dtype;
    that regression made the results CSV unsortable."""
    ref = _reference([200.1234])
    preds = _predictions([
        ("A", 200.1234, np.full(D, 1.0)),
        ("B", 999.9999, np.full(D, 1.0)),
    ])

    out = match_predictions_to_reference_without_smiles(preds, ref, 2, "without_smiles", "+")

    assert out["Top_1_cosine"].dtype.kind == "f"
    out.sort_values("Top_1_cosine")


# ---------------------------------------------------------------------------
# The default must be the fixed behaviour. Shipping the correct code path
# behind a flag that defaults to the broken one is what happened in 1.1.1.
# ---------------------------------------------------------------------------
def test_default_tolerance_is_a_real_window_not_the_legacy_bucket():
    assert DEFAULT_PRECURSOR_TOLERANCE_PPM > 0


def test_matcher_defaults_to_the_window_when_no_tolerance_is_passed():
    mass = 250.0806                       # only found by a window
    ref = _reference([mass])
    preds = _predictions([("QUERY", mass, np.full(D, 1.0))])

    out = match_predictions_to_reference_without_smiles(preds, ref, 1, "without_smiles", "+")
    assert out["Top_1_SMILE"].iloc[0] == "SMILES_0"


def test_batch_config_defaults_to_a_real_window():
    """chemembed_by_file reads precursor_tolerance_ppm from the config; the default
    applied there must match the library default."""
    cfg = {}
    cfg.setdefault("precursor_tolerance_ppm", 5.0)
    assert cfg["precursor_tolerance_ppm"] == DEFAULT_PRECURSOR_TOLERANCE_PPM


def test_proton_mass_constant_is_shared():
    assert PROTON_MASS == pytest.approx(1.007276)
