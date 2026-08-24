"""Fast regression tests. None of these need the models or the reference database.

Each test corresponds to a defect that reached a release or a release candidate,
so a failure here means a bug we have already shipped once is back.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# DataLoader collates a per-item string into a one-element sequence. At
# batch_size=1 that reached the results CSV as the literal text "('ABC',)",
# so the column could not be joined on. Fixed in 1.0.1.
# --------------------------------------------------------------------------
def test_unbatch_unwraps_collated_strings():
    from chemembed.model_utils import _unbatch

    assert _unbatch(["ABC"]) == "ABC"
    assert _unbatch(("ABC",)) == "ABC"
    assert _unbatch("ABC") == "ABC"


def test_unbatch_leaves_real_sequences_alone():
    from chemembed.model_utils import _unbatch

    assert _unbatch(["A", "B"]) == ["A", "B"]
    assert _unbatch([]) == []


# --------------------------------------------------------------------------
# Cosine columns were initialised to the string "NA", which made the column
# object dtype: sorting raised TypeError and numeric work was unreliable.
# They must come back as a float dtype.
# --------------------------------------------------------------------------
def test_cosine_columns_are_numeric():
    from math import nan

    df = pd.DataFrame({"Top Min_cosine": [[0.91, 0.72], [0.55]]})
    for c in ("Top_1_cosine", "Top_2_cosine"):
        df[c] = nan
    for i in range(len(df)):
        for c, v in zip(["Top_1_cosine", "Top_2_cosine"], df["Top Min_cosine"].iloc[i]):
            df.at[i, c] = v

    assert df["Top_1_cosine"].dtype.kind == "f"
    df.sort_values("Top_2_cosine")            # raised TypeError when object dtype


# --------------------------------------------------------------------------
# MSP parsing
# --------------------------------------------------------------------------
MSP = textwrap.dedent(
    """\
    Name: TESTCOMPOUND01
    Precursor: 524.371
    Adduct: [M+H]+
    138.0662 100.0
    110.0713 45.5

    Name: TESTCOMPOUND02
    Precursor: 209.081
    Adduct: [M+H]+
    147.0441 88.0
    """
)


def test_msp_parser_reads_both_spectra(tmp_path):
    from chemembed.data_processing import msp_to_dataframe_without_smiles

    p = tmp_path / "spectra.msp"
    p.write_text(MSP)
    df = msp_to_dataframe_without_smiles(str(p))

    assert len(df) == 2
    assert df["Unique_ID"].tolist() == ["TESTCOMPOUND01", "TESTCOMPOUND02"]
    assert df["Precursor"].tolist() == [524.371, 209.081]
    assert df["spectra"].iloc[0] == [(138.0662, 100.0), (110.0713, 45.5)]
    assert df["Adduct"].iloc[0] == "[M+H]+"


# --------------------------------------------------------------------------
# The CLI required both polarities' files even when only one was used, so a
# valid positive-mode invocation was rejected by argparse. Fixed pre-1.0.0.
# It must now get past argument parsing and fail on the missing input file.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "adduct,msp_flag,model_flag",
    [("+", "--msp_file_positive", "--model_path_positive"),
     ("-", "--msp_file_negative", "--model_path_negative")],
)
def test_cli_accepts_single_polarity(adduct, msp_flag, model_flag, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "chemembed.cli",
         "--input_file_type", "without_smiles", "--adduct", adduct,
         msp_flag, str(tmp_path / "missing.msp"),
         model_flag, str(tmp_path / "missing.bin"),
         "--reference_database", str(tmp_path / "missing.pkl")],
        capture_output=True, text=True,
    )
    combined = r.stdout + r.stderr
    # argparse rejection would say "the following arguments are required"
    assert "the following arguments are required" not in combined, combined
    # getting as far as opening the file is the success condition here
    assert "FileNotFoundError" in combined or "No such file" in combined, combined


def test_cli_reports_missing_file_for_chosen_polarity(tmp_path):
    """Omitting the flag the chosen adduct needs must fail clearly, not silently."""
    r = subprocess.run(
        [sys.executable, "-m", "chemembed.cli",
         "--input_file_type", "without_smiles", "--adduct", "+",
         "--reference_database", str(tmp_path / "missing.pkl")],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "msp_file_positive" in (r.stdout + r.stderr)


# --------------------------------------------------------------------------
# The reference reader must accept parquet and pickle, sniffing by extension.
# --------------------------------------------------------------------------
def test_reference_reader_roundtrips_pickle(tmp_path):
    from chemembed.reference_utils import _read_reference_any_format

    df = pd.DataFrame({
        "smile": ["CCO", "CCN"],
        "inchikey": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
        "up_molvec": [np.zeros(300, dtype=np.float32)] * 2,
    })
    p = tmp_path / "ref.pkl"
    df.to_pickle(p)
    back = _read_reference_any_format(str(p))
    assert len(back) == 2
    assert list(back.columns) == list(df.columns)


def test_stored_precursormz_is_rejected_when_wrong():
    """The fast path must validate, not assume: a bogus column has to fail."""
    from chemembed.reference_utils import _stored_precursormz_is_trustworthy

    bad = pd.DataFrame({"smile": ["CCO", "CCN"], "Precursormz": [1.0, 2.0]})
    assert _stored_precursormz_is_trustworthy(bad, "+") is False


# --------------------------------------------------------------------------
# Under the flat layout there was no importable `chemembed` module at all,
# despite that being the install name. The src/ layout fixes that.
# --------------------------------------------------------------------------
def test_package_is_importable_under_its_install_name():
    import chemembed

    assert hasattr(chemembed, "__version__")


def test_public_entry_point_is_reachable_from_the_package():
    import chemembed

    assert callable(chemembed.run)


# --------------------------------------------------------------------------
# The version was declared in three places and drifted: 1.1.1 shipped with
# __init__.py still saying "1.1.0". __version__ now comes from the installed
# distribution metadata, so it cannot disagree with pyproject.toml.
# --------------------------------------------------------------------------
def test_version_matches_the_installed_distribution():
    from importlib.metadata import version

    import chemembed

    assert chemembed.__version__ == version("chemembed")


def test_citation_file_version_matches_the_package(tmp_path):
    """CITATION.cff carries its own version field and has to be bumped in step."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    cff = root / "CITATION.cff"
    pyproject = root / "pyproject.toml"
    if not cff.exists() or not pyproject.exists():
        pytest.skip("running against an installed package, not a source checkout")

    cff_version = re.search(r"^version:\s*(\S+)", cff.read_text(), re.M).group(1)
    proj_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert cff_version == proj_version, (
        f"CITATION.cff says {cff_version}, pyproject.toml says {proj_version}"
    )
