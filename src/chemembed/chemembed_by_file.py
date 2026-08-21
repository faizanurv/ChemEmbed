#!/usr/bin/env python3
"""
chemembed_by_file.py  —  ChemEmbed annotation module (FAISS GPU edition)
=========================================================================
Iterates over all samples, reads the GNPS/SIRIUS MGF file from each
sample's polarity folder, and produces ChemEmbed predictions.

Speed improvements over the original code
------------------------------------------
1. Reference DB loaded ONCE before the sample loop (was reloaded per sample).
   RDKit Precursormz computation: done once, not N_samples x 5.5M times.
2. FAISS GPU index built once — all queries for a sample searched in ONE GPU
   call (~1 ms regardless of DB size).
3. Precursor m/z post-filter applied to FAISS results to maintain biological
   correctness (exact 3-decimal match, same as original ChemEmbed logic).
   Fallback dict index handles sparse masses that fall outside faiss_k hits.
4. CNN inference runs in batches (batch_size=32) on GPU instead of batch=1
   on CPU — typically 20-50x faster.
5. predict_without_smiles rebuilt to support batch_size > 1.

Output per sample:
    <sample>/<pol>/chemembed_annotation/
        - converted_spectra.msp
        - preprocessed_data.pkl
        - chemembed_results.csv

Usage:
    python chemembed_by_file.py
    python chemembed_by_file.py --config /path/to/chemembed_config.yml

Optional YAML keys (all have defaults):
    batch_size:   32      # CNN inference batch size
    num_workers:  0       # DataLoader workers
    faiss_k:      200     # FAISS candidates before precursor post-filter
    recompute:    false
"""

import argparse
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ChemEmbed annotation — FAISS GPU edition"
    )
    parser.add_argument(
        "--config", "-c",
        default="chemembed_config.yml",
        help="Path to config YAML (default: chemembed_config.yml)",
    )
    return parser.parse_args()


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    required = [
        "treated_data_path", "ionization",
        "model_path_positive", "model_path_negative", "reference_database",
    ]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if cfg["ionization"] not in ("pos", "neg"):
        raise ValueError(f"ionization must be 'pos' or 'neg', got: {cfg['ionization']}")

    # Defaults
    cfg.setdefault("tolerance",           0.01)
    cfg.setdefault("max_mz",              700)
    cfg.setdefault("resolution",          0.01)
    cfg.setdefault("intensity_threshold", 1)
    cfg.setdefault("top_n_candidates",    5)
    cfg.setdefault("recompute",           False)
    cfg.setdefault("batch_size",          32)   # CNN batch size
    cfg.setdefault("num_workers",         0)    # DataLoader workers
    cfg.setdefault("faiss_k",             200)  # FAISS candidates before precursor filter

    # chemembed_root is vestigial: the modules now ship inside this package
    # and are imported directly. Still honoured so existing configs keep working.
    root = cfg.get("chemembed_root")
    if root and root not in sys.path:
        sys.path.insert(0, root)

    return cfg


# ==============================================================================
# MGF PARSING
# ==============================================================================

def parse_mgf(mgf_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse an MGF file. Returns (spectra_list, id_strategy_description)."""
    spectra: List[Dict[str, Any]] = []
    raw_id_field_used: Optional[str] = None

    with open(mgf_path, "r", encoding="utf-8", errors="replace") as fh:
        in_spectrum = False
        current: Dict[str, Any] = {}
        peaks: List[Tuple[float, float]] = []

        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.upper() == "BEGIN IONS":
                in_spectrum = True
                current = {"metadata": {}}
                peaks = []
                continue

            if line.upper() == "END IONS":
                current["peaks"] = peaks
                if current.get("mslevel", "2") == "2":
                    spectra.append(current)
                in_spectrum = False
                continue

            if not in_spectrum:
                continue

            if "=" in line and not re.match(r"^[\d.]+\s+[\d.]+", line):
                key, _, value = line.partition("=")
                key   = key.strip().upper()
                value = value.strip()

                if key == "PEPMASS":
                    parts = value.split()
                    try:
                        current["pepmass"] = float(parts[0])
                    except (ValueError, IndexError):
                        current["pepmass"] = None
                elif key in ("RTINSECONDS", "RT"):
                    try:
                        current["rt"] = float(value)
                    except ValueError:
                        current["rt"] = None
                elif key == "CHARGE":
                    current["charge"] = value
                elif key == "ION":
                    current["ion"] = value
                elif key == "MSLEVEL":
                    current["mslevel"] = value
                elif key in ("FEATURE_ID", "FEATUREID"):
                    current["feature_id_raw"] = value
                    raw_id_field_used = key
                elif key == "SCANS":
                    current.setdefault("scans", value)
                else:
                    current["metadata"][key] = value
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    if any("feature_id_raw" in s for s in spectra):
        id_strategy = f"MGF field '{raw_id_field_used}'"
        for i, s in enumerate(spectra):
            s["feature_id"] = s.pop(
                "feature_id_raw", s.get("scans", f"feature_{i+1:04d}")
            )
    elif any("scans" in s for s in spectra):
        id_strategy = "MGF field 'SCANS'"
        for i, s in enumerate(spectra):
            s["feature_id"] = s.get("scans", f"feature_{i+1:04d}")
    else:
        id_strategy = "sequential index"
        for i, s in enumerate(spectra):
            s["feature_id"] = f"feature_{i+1:04d}"

    return spectra, id_strategy


# ==============================================================================
# MGF -> MSP CONVERSION
# ==============================================================================

def spectrum_to_msp(spectrum: Dict[str, Any], adduct_string: str) -> str:
    """
    Convert one parsed MGF spectrum to an MSP block.
    Field names match exactly what ChemEmbed's parser expects (case-sensitive).
    """
    lines: List[str] = []
    fid  = spectrum["feature_id"]
    name = f"feature_{fid}" if str(fid).isdigit() else str(fid)

    lines.append(f"Name: {name}")
    lines.append(f"FEATURE_ID: {fid}")

    pepmass = spectrum.get("pepmass")
    lines.append(f"Precursor: {pepmass:.6f}" if pepmass is not None else "Precursor: 0.000000")
    lines.append(f"Adduct: {adduct_string}")

    rt = spectrum.get("rt")
    if rt is not None:
        lines.append(f"RTINSECONDS: {rt:.4f}")

    skip_keys = {"SMILES", "INCHI", "INCHIKEY", "FORMULA"}
    for k, v in spectrum.get("metadata", {}).items():
        if k.upper() not in skip_keys:
            lines.append(f"{k}: {v}")

    peaks = spectrum["peaks"]
    lines.append(f"Num Peaks: {len(peaks)}")
    for mz, intensity in peaks:
        lines.append(f"{mz:.6f} {intensity:.4f}")

    return "\n".join(lines)


def is_compatible_adduct(spectrum: Dict[str, Any], ionization: str) -> bool:
    """
    Return True if the spectrum should be processed as the canonical ChemEmbed
    adduct ([M+H]+ for pos, [M-H]- for neg).

    Accepted cases:
      - ION field is the canonical adduct alone.
      - ION field is unspecified (fall back to charge / default).
      - ION field lists MULTIPLE adducts (e.g. "[M+H]+ [M+Na]+ [M+K]+ [M+NH4]+
        [M+H-H2O]+") AND the canonical adduct is present ANYWHERE in the list.
        In that case we default to the canonical [M+H]+ / [M-H]- regardless of
        order, since ChemEmbed only models that adduct. (Previously such rows were
        neglected unless the canonical adduct happened to be listed first.)
    """
    allowed = "[M+H]+" if ionization == "pos" else "[M-H]-"
    ion = spectrum.get("ion", "")
    if ion:
        tokens = re.findall(r"\[[^\]]+\]\d*[+-]", ion)
        if len(tokens) > 1:
            # Multiple adducts listed: accept and default to canonical if present.
            return allowed in tokens
        # Single (or unparseable) adduct: require the canonical adduct.
        return allowed in ion
    charge = str(spectrum.get("charge", "")).strip()
    if charge:
        return charge in ("1", "1+", "1-")
    return True


def merge_spectra_by_feature(
    spectra: List[Dict[str, Any]],
    ionization: str,
    bin_size: float = 0.01,
) -> List[Dict[str, Any]]:
    """Group by feature_id, filter adduct, merge peaks via binary union."""
    from collections import defaultdict
    groups: Dict[str, list] = defaultdict(list)
    for s in spectra:
        if s.get("peaks") and is_compatible_adduct(s, ionization):
            groups[s.get("feature_id", "unknown")].append(s)

    merged = []
    for fid, group in groups.items():
        occupied = set()
        for s in group:
            for mz, _ in s["peaks"]:
                occupied.add(round(mz / bin_size))
        if not occupied:
            continue
        base = group[0].copy()
        base["peaks"] = sorted(
            [(b * bin_size, 1.0) for b in occupied], key=lambda x: x[0]
        )
        base["feature_id"] = fid
        merged.append(base)
    return merged


def write_msp(
    spectra: List[Dict[str, Any]],
    msp_path: str,
    adduct_string: str,
    ionization: str,
) -> Tuple[int, int, int]:
    """Merge, filter, and write MSP. Returns (written, skipped_no_peaks, skipped_adduct)."""
    merged     = merge_spectra_by_feature(spectra, ionization)
    n_no_peaks = sum(1 for s in spectra if not s.get("peaks"))
    n_incompat = sum(
        1 for s in spectra
        if s.get("peaks") and not is_compatible_adduct(s, ionization)
    )
    n_written = 0
    with open(msp_path, "w", encoding="utf-8") as fh:
        for s in merged:
            if not s["peaks"]:
                continue
            fh.write(spectrum_to_msp(s, adduct_string))
            fh.write("\n\n")
            n_written += 1
    return n_written, n_no_peaks, n_incompat


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def _truncate_mass(mz: float) -> float:
    """Floor-truncate to 3 decimal places — identical to ChemEmbed's original logic."""
    return int(float(mz) * 1000) / 1000.0


def _unpack_vec(v) -> np.ndarray:
    """
    Extract a flat float32 vector from any storage format used in ChemEmbed DBs:
        (300,)        flat array
        (1, 300)      nested array  <- what your DB uses
        (300, 1)      column vector
        object array  wrapping any of the above
    """
    try:
        return np.asarray(v, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return np.asarray(np.asarray(v).flat[0], dtype=np.float32).ravel()


def _unwrap_uid(uid) -> str:
    """Unwrap ChemEmbed tuple Unique_ID e.g. ('feature_42',) -> 'feature_42'."""
    if isinstance(uid, tuple):
        return str(uid[0]) if uid else ""
    return str(uid)


# ==============================================================================
# REFERENCE DATABASE — ONE-TIME SETUP (FAISS GPU)
# ==============================================================================

def load_reference_cache(reference_database: str, adduct: str,
                         build_tolerance_index: bool = False) -> Dict[str, Any]:
    """
    Load the reference DB once and build:

    1. FAISS GPU IndexFlatIP over all R normalised embeddings.
       A single index.search(Q_mat, k) retrieves top-k for ALL Q queries
       simultaneously on GPU (~1ms regardless of R or Q).

    2. Metadata arrays (inchikey, smile, truncated precursor m/z) aligned
       with FAISS index row order so result indices map directly to metadata.

    3. Fallback dict {truncated_mz: {emb_norm, inchikey, smile}} for the rare
       case where a query's faiss_k hits don't contain enough precursor matches.

    Notes
    -----
    - load_reference_database_without_smiles (from reference_utils) computes
      Precursormz via RDKit from SMILES. This is expensive (~1-2 min for 5.5M).
      By loading once here it is never repeated across samples.
    - The FAISS index stores L2-normalised vectors so inner product == cosine.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss not installed.\n"
            "  GPU:  conda install -c pytorch faiss-gpu\n"
            "  CPU:  conda install -c pytorch faiss-cpu"
        )

    try:
        from .reference_utils import load_reference_database_without_smiles
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"ChemEmbed import failed: {exc}\n"
            "The ChemEmbed package appears to be installed incompletely."
        )

    # ── Load raw reference DataFrame ─────────────────────────────────────────
    print("Loading reference database (one-time)...")
    t0     = time.perf_counter()
    ref_df = load_reference_database_without_smiles(reference_database, adduct)
    R      = len(ref_df)
    print(f"  {R:,} entries loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  Columns: {list(ref_df.columns)}")

    # ── Identify columns ──────────────────────────────────────────────────────
    # Embedding column: first column whose first cell is array-like
    emb_col = None
    for col in ref_df.columns:
        val = ref_df[col].iloc[0]
        if hasattr(val, "__len__") and not isinstance(val, str):
            emb_col = col
            break
    if emb_col is None:
        raise ValueError(f"No embedding column found. Columns: {list(ref_df.columns)}")

    ik_col   = next((c for c in ("inchikey", "InChIKey", "INCHIKEY", "up_inchikey")  if c in ref_df.columns), None)
    sm_col   = next((c for c in ("smile",    "SMILE",    "smiles", "SMILES") if c in ref_df.columns), None)
    prec_col = "Precursormz"   # always present after load_reference_database_without_smiles

    print(f"  emb_col={emb_col!r}  ik_col={ik_col!r}  sm_col={sm_col!r}  prec_col={prec_col!r}")

    # Drop rows where precursor m/z is NaN — prevents crash in _truncate_mass
    n_before = len(ref_df)
    ref_df = ref_df.dropna(subset=[prec_col]).reset_index(drop=True)
    n_dropped = n_before - len(ref_df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} entries with NaN {prec_col} (kept {len(ref_df):,})")

    # ── Stack and L2-normalise all embeddings ─────────────────────────────────
    print("  Stacking embeddings...")
    t1      = time.perf_counter()
    all_emb = np.stack([_unpack_vec(v) for v in ref_df[emb_col].values]).astype(np.float32)
    D       = all_emb.shape[1]
    norms   = np.linalg.norm(all_emb, axis=1, keepdims=True)
    all_emb /= (norms + 1e-9)
    all_emb  = np.ascontiguousarray(all_emb)
    print(f"  Embedding matrix: ({R:,}, {D}) built in {time.perf_counter()-t1:.1f}s")

    # Truncated precursor array aligned with all_emb (original row order).
    # This is what the ORIGINAL exact-bucket filter uses; kept for backward compatibility.
    prec_trunc = np.array(
        [_truncate_mass(v) for v in ref_df[prec_col].values],
        dtype=np.float64,
    )

    # Untruncated precursor, aligned with all_emb, for the OPTIONAL ppm-tolerance filter.
    # The original code buckets precursors to 3 dp and compares by equality, which discards
    # ~half of correct candidates on real data (reference rounds, query floors, `==` fails
    # whenever the 4th decimal >= 5). A ppm window compares the real masses instead, so we
    # keep the full-precision value here. Built only when asked, to spare the production
    # (exact-bucket) path the extra memory.
    prec_exact = ref_df[prec_col].values.astype(np.float64)

    # Metadata arrays aligned with all_emb
    ik_arr = ref_df[ik_col].values if ik_col else None
    sm_arr = ref_df[sm_col].values if sm_col else None

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("  Building FAISS index...")
    t2      = time.perf_counter()
    n_gpus  = faiss.get_num_gpus()
    cpu_idx = faiss.IndexFlatIP(D)
    cpu_idx.add(all_emb)

    if n_gpus > 0:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
        dev   = f"GPU:0 (of {n_gpus} available)"
    else:
        index = cpu_idx
        dev   = "CPU (no GPU detected — install faiss-gpu for full speed)"

    print(f"  FAISS: {index.ntotal:,} vectors | {dev} | {time.perf_counter()-t2:.1f}s")

    # ── Build fallback dict index ─────────────────────────────────────────────
    # Groups pre-normalised embeddings by truncated precursor mass.
    # Used when faiss_k hits don't contain enough precursor-matching candidates.
    print("  Building fallback dict index...")
    t3           = time.perf_counter()
    sort_idx     = np.argsort(prec_trunc, kind="stable")
    trunc_sorted = prec_trunc[sort_idx]
    emb_sorted   = np.ascontiguousarray(all_emb[sort_idx])
    ik_sorted    = ik_arr[sort_idx] if ik_arr is not None else None
    sm_sorted    = sm_arr[sort_idx] if sm_arr is not None else None

    unique_masses, g_starts, g_counts = np.unique(
        trunc_sorted, return_index=True, return_counts=True
    )

    fallback: Dict[float, Dict[str, Any]] = {}
    for mass, start, count in zip(unique_masses, g_starts, g_counts):
        end   = int(start) + int(count)
        entry: Dict[str, Any] = {"emb_norm": emb_sorted[start:end]}   # zero-copy slice
        if ik_sorted is not None:
            entry["inchikey"] = ik_sorted[start:end].tolist()
        if sm_sorted is not None:
            entry["smile"]    = sm_sorted[start:end].tolist()
        fallback[float(mass)] = entry

    print(f"  Fallback: {len(fallback):,} mass groups | {time.perf_counter()-t3:.1f}s")

    cache = {
        "faiss_index": index,       # FAISS GPU/CPU index
        "prec_trunc":  prec_trunc,  # (R,) truncated precursor aligned with FAISS rows
        "prec_exact":  prec_exact,  # (R,) untruncated precursor aligned with FAISS rows
        "ik_arr":      ik_arr,      # (R,) inchikey aligned with FAISS rows
        "sm_arr":      sm_arr,      # (R,) smile    aligned with FAISS rows
        "fallback":    fallback,    # {trunc_mz: {emb_norm, inchikey, smile}}
        "ik_col":      ik_col,
        "sm_col":      sm_col,
        "n_total":     R,
        "D":           D,
    }

    # ── OPTIONAL: exact-mass window index for the ppm-tolerance filter ─────────
    # The tolerance fallback needs to gather all reference rows within a mass window, so
    # it needs the embeddings sorted by EXACT mass (the truncated fallback dict groups by
    # 3-dp bucket, which is the wrong granularity for a window query) plus the full
    # embedding matrix to gather from. Built only when a ppm tolerance is configured, so
    # the default path pays neither the ~6.6GB to retain all_emb nor the sort.
    if build_tolerance_index:
        print("  Building exact-mass window index (ppm-tolerance mode)...")
        t4 = time.perf_counter()
        exact_order = np.argsort(prec_exact, kind="stable")
        cache["all_emb"]           = all_emb                    # (R,D) normalised, gather source
        cache["exact_order"]       = exact_order               # (R,) rows sorted by exact mass
        cache["prec_exact_sorted"] = prec_exact[exact_order]   # (R,) ascending, for searchsorted
        print(f"  Exact-mass index ready | {time.perf_counter()-t4:.1f}s")

    print(f"  Reference cache ready.\n")
    return cache


# ==============================================================================
# FAISS MATCHING  (GPU + precursor post-filter)
# ==============================================================================

def match_with_faiss(
    prediction_df: pd.DataFrame,
    ref_cache: Dict[str, Any],
    top_n: int,
    faiss_k: int = 200,
    tol_ppm: float = 0.0,
) -> pd.DataFrame:
    """
    Match all Q query embeddings against the reference database.

    Algorithm
    ---------
    1. Stack all Q query vectors into (Q, D) matrix — one GPU transfer.
    2. faiss_index.search(Q_mat, faiss_k) — single GPU call returning
       (Q, faiss_k) scores and indices for all queries simultaneously.
    3. Per query: keep only FAISS hits whose precursor m/z matches the query, then
       4. return top_n by cosine; if fewer than top_n survive, fall back to a direct
          mass-indexed search + CPU matmul.

    Precursor matching mode (Parameters -> tol_ppm)
    -----------------------------------------------
    * tol_ppm == 0  (default): the ORIGINAL exact-bucket filter — reference and query
      precursors are each reduced to 3 decimals and compared by equality. This is what
      shipped, and it is preserved bit-for-bit for the production pipeline. NOTE it is
      also buggy: the reference rounds and the query floors, so the `==` fails whenever
      the 4th decimal is >= 5, discarding ~half of correct candidates on exact data.
    * tol_ppm > 0: a real mass WINDOW — keep any reference within tol_ppm of the query's
      measured precursor (|ref - query| <= query * tol_ppm / 1e6). This is the fix; it
      requires the exact-mass index (load_reference_cache(..., build_tolerance_index=True)).

    Parameters
    ----------
    faiss_k : int
        How many FAISS candidates to retrieve per query before post-filtering.
    tol_ppm : float
        Precursor mass tolerance in ppm. 0 selects the original exact-bucket behaviour.
    """
    faiss_index = ref_cache["faiss_index"]
    prec_trunc  = ref_cache["prec_trunc"]
    ik_arr      = ref_cache["ik_arr"]
    sm_arr      = ref_cache["sm_arr"]
    fallback    = ref_cache["fallback"]
    ik_col      = ref_cache["ik_col"]
    sm_col      = ref_cache["sm_col"]

    tol_ppm  = float(tol_ppm or 0.0)
    use_tol  = tol_ppm > 0.0
    if use_tol:
        if "prec_exact" not in ref_cache or "exact_order" not in ref_cache:
            raise ValueError(
                "tol_ppm > 0 requires the exact-mass index; call "
                "load_reference_cache(..., build_tolerance_index=True).")
        prec_exact        = ref_cache["prec_exact"]
        all_emb           = ref_cache["all_emb"]
        exact_order       = ref_cache["exact_order"]
        prec_exact_sorted = ref_cache["prec_exact_sorted"]
        print(f"    [match] precursor filter: ±{tol_ppm:g} ppm window")
    else:
        print(f"    [match] precursor filter: exact 3-dp bucket (original)")

    data_test = prediction_df.reset_index(drop=True)
    Q         = len(data_test)

    # ── Build (Q, D) query matrix ─────────────────────────────────────────────
    raw_vecs = [_unpack_vec(v) for v in data_test["tree_out"].tolist()]
    Q_mat    = np.stack(raw_vecs).astype(np.float32)
    Q_mat   /= (np.linalg.norm(Q_mat, axis=1, keepdims=True) + 1e-9)
    Q_mat    = np.ascontiguousarray(Q_mat)

    # Query precursor, in both forms: truncated (exact-bucket path) and full precision
    # (ppm-window path). The window path must NOT floor the query — flooring is half of
    # the bug it exists to fix.
    query_trunc = np.array(
        [_truncate_mass(float(v)) for v in data_test["Precursor"].values],
        dtype=np.float64,
    )
    query_exact = data_test["Precursor"].to_numpy(dtype=np.float64)

    # ── Single FAISS search — ALL Q queries at once ───────────────────────────
    k_search            = min(faiss_k, ref_cache["n_total"])
    scores_mat, idx_mat = faiss_index.search(Q_mat, k_search)  # (Q, k_search) each

    # ── Per-query post-filter + optional fallback ─────────────────────────────
    final_uid      = []
    final_cosine   = []
    final_inchikey = []
    final_smile    = []
    n_fallback     = 0

    for i in range(Q):
        uid  = _unwrap_uid(data_test["Unique_ID"].iloc[i])

        cand_idx    = idx_mat[i]                                       # (k_search,)
        cand_scores = scores_mat[i]                                    # (k_search,) desc
        valid       = cand_idx >= 0

        if use_tol:
            qx   = query_exact[i]
            da   = qx * tol_ppm / 1e6
            mask = valid & (np.abs(prec_exact[cand_idx] - qx) <= da)
        else:
            mask = valid & (prec_trunc[cand_idx] == query_trunc[i])
        f_idx    = cand_idx[mask]
        f_scores = cand_scores[mask]                                   # already sorted desc

        if len(f_scores) >= top_n:
            # FAISS path — enough precursor-matched candidates in the top-k
            take       = top_n
            top_scores = f_scores[:take].tolist()
            top_ik     = [ik_arr[j] for j in f_idx[:take]] if ik_col else ["NA"] * take
            top_sm     = [sm_arr[j] for j in f_idx[:take]] if sm_col else ["NA"] * take

        else:
            # Fallback: gather ALL precursor-matching references, then CPU matmul. The
            # FAISS top-k missed some because the true match ranked below k on cosine.
            n_fallback += 1
            if use_tol:
                # Window query over exact masses via binary search on the sorted array.
                lo   = np.searchsorted(prec_exact_sorted, query_exact[i] - da, side="left")
                hi   = np.searchsorted(prec_exact_sorted, query_exact[i] + da, side="right")
                rows = exact_order[lo:hi]                              # original row indices
                if len(rows) == 0:
                    continue
                emb_g   = all_emb[rows]                                # (N, D) gather
                sc      = emb_g @ Q_mat[i]
                n_g     = len(sc)
                take    = min(top_n, n_g)
                if take == n_g:
                    top_local = np.argsort(-sc)[:take]
                else:
                    top_local = np.argpartition(sc, -take)[-take:]
                    top_local = top_local[np.argsort(-sc[top_local])]
                sel        = rows[top_local]
                top_scores = sc[top_local].tolist()
                top_ik     = [ik_arr[j] for j in sel] if ik_col else ["NA"] * take
                top_sm     = [sm_arr[j] for j in sel] if sm_col else ["NA"] * take
            else:
                group = fallback.get(query_trunc[i])
                if group is None:
                    continue
                emb_g  = group["emb_norm"]                             # (N_group, D)
                sc     = emb_g @ Q_mat[i]
                n_g    = len(sc)
                take   = min(top_n, n_g)
                if take == n_g:
                    top_local = np.argsort(-sc)[:take]
                else:
                    top_local = np.argpartition(sc, -take)[-take:]
                    top_local = top_local[np.argsort(-sc[top_local])]
                top_scores = sc[top_local].tolist()
                top_ik     = [group["inchikey"][j] for j in top_local] if "inchikey" in group else ["NA"] * take
                top_sm     = [group["smile"][j]    for j in top_local] if "smile"    in group else ["NA"] * take

        final_uid.append(uid)
        final_cosine.append(top_scores)
        final_inchikey.append(top_ik)
        final_smile.append(top_sm)

    if n_fallback > 0:
        print(f"    [match] Fallback used: {n_fallback}/{Q} queries "
              f"(consider increasing faiss_k from current={faiss_k})")

    # ── Assemble results DataFrame (same column format as original ChemEmbed) ─
    rows = []
    for i in range(len(final_uid)):
        row: Dict[str, Any] = {"Unique_ID": final_uid[i]}
        for k in range(top_n):
            row[f"Top_{k+1}_cosine"]   = final_cosine[i][k]   if k < len(final_cosine[i])   else "NA"
            row[f"Top_{k+1}_SMILE"]    = final_smile[i][k]    if k < len(final_smile[i])    else "NA"
            row[f"Top_{k+1}_InChIKey"] = final_inchikey[i][k] if k < len(final_inchikey[i]) else "NA"
        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# CNN INFERENCE  (batched, GPU-aware)
# ==============================================================================

def predict_batched(
    model: Any,
    test_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run CNN inference in batches on the given device.

    Replaces ChemEmbed's original predict_without_smiles which used batch_size=1
    on CPU. Supports batch_size > 1 on GPU for 20-50x speedup.

    Returns a DataFrame with columns matching ChemEmbed's original schema:
        Precursor, tree_out, len_frag, Unique_ID
    """
    model.eval()
    model.to(device)

    records = []
    with torch.no_grad():
        for inputs in test_loader:
            # inputs[0]: spectral tensor  (B, ...)
            # inputs[1]: precursor m/z    (B,)
            # inputs[2]: len_frag         (B,)
            # inputs[3]: Unique_ID        list of B strings/tuples
            batch_tensor = inputs[0].to(device)
            outputs      = model(batch_tensor).detach().cpu().numpy()  # (B, D)
            B            = outputs.shape[0]

            for b in range(B):
                precursor = inputs[1][b].item() if hasattr(inputs[1][b], "item") else float(inputs[1][b])
                len_frag  = inputs[2][b].item() if hasattr(inputs[2][b], "item") else int(inputs[2][b])
                # Unique_ID may be a list (batched strings) or a single tensor
                if isinstance(inputs[3], (list, tuple)):
                    unique_id = inputs[3][b]
                else:
                    unique_id = inputs[3]
                records.append({
                    "Precursor":  precursor,
                    "tree_out":   outputs[b:b+1],   # keep (1, D) shape to match original schema
                    "len_frag":   len_frag,
                    "Unique_ID":  unique_id,
                })

    return pd.DataFrame(records)


# ==============================================================================
# CHEMEMBED PIPELINE  (per sample)
# ==============================================================================

def run_chemembed(
    cfg: Dict[str, Any],
    ref_cache: Dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """
    Full ChemEmbed pipeline for one sample.

    Steps 1-4 (MSP -> preprocess -> vectorise -> pkl) are unchanged.
    Step 5: batched CNN inference on GPU.
    Step 6: FAISS GPU search + precursor post-filter.

    Parameters
    ----------
    cfg       : per-sample runtime config dict
    ref_cache : pre-built reference cache from load_reference_cache()
    device    : torch device (cuda or cpu)
    """
    try:
        from .data_processing import (
            msp_to_dataframe_without_smiles,
            preprocess_spectra_without_smiles,
            process_data_without_smiles,
        )
        from .model_utils import load_model
        from .data_loaders import spectra_inference_dataset_loader as data_loader_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"ChemEmbed import failed: {exc}\n"
            "The ChemEmbed package appears to be installed incompletely."
        )

    adduct = cfg["adduct"]

    print("    [1/5] Loading MSP...")
    msp_df = msp_to_dataframe_without_smiles(cfg["msp_output"])

    print("    [2/5] Preprocessing spectra...")
    norm_df = preprocess_spectra_without_smiles(msp_df, cfg["intensity_threshold"])

    print("    [3/5] Building spectral vectors...")
    final_df = process_data_without_smiles(
        norm_df, cfg["tolerance"], cfg["resolution"], cfg["max_mz"]
    )

    print("    [4/5] Saving preprocessed data...")
    final_df.to_pickle(cfg["preprocessed_data"])

    # ── Batched CNN inference on GPU ──────────────────────────────────────────
    batch_size  = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 0)
    print(f"    [5/5] CNN inference | device={device} | batch_size={batch_size}...")

    test_dataset = data_loader_module.class_ls(cfg["preprocessed_data"])
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        drop_last   = False,   # False: do not discard the last partial batch
        shuffle     = False,
        num_workers = num_workers,
    )

    model_path = cfg["model_path_positive"] if adduct == "+" else cfg["model_path_negative"]
    model_cnn  = load_model(model_path)

    t_cnn         = time.perf_counter()
    prediction_df = predict_batched(model_cnn, test_loader, device)
    cnn_elapsed   = time.perf_counter() - t_cnn
    print(f"    [5/5] CNN done: {len(prediction_df)} spectra in {cnn_elapsed:.1f}s "
          f"({cnn_elapsed/max(len(prediction_df),1)*1000:.0f}ms/spectrum)")

    # ── FAISS GPU matching ────────────────────────────────────────────────────
    print("    [match] FAISS GPU matching...")
    t_match    = time.perf_counter()
    results_df = match_with_faiss(
        prediction_df = prediction_df,
        ref_cache     = ref_cache,
        top_n         = cfg["top_n_candidates"],
        faiss_k       = cfg.get("faiss_k", 200),
        tol_ppm       = float(cfg.get("precursor_tolerance_ppm", 0) or 0),
    )
    match_elapsed = time.perf_counter() - t_match
    print(f"    [match] Done: {len(prediction_df)} queries in {match_elapsed*1000:.0f}ms")

    return results_df


# ==============================================================================
# POST-PROCESSING
# ==============================================================================

def clean_unique_id(uid_value: Any) -> str:
    """
    Normalise Unique_ID back to the original feature_id.
    Handles stringified tuples "('feature_42',)", plain strings, and bare tuples.
    """
    s = str(uid_value).strip()
    m = re.match(r"^\(['\"](.+?)['\"],?\s*\)$", s)
    if m:
        s = m.group(1)
    n = re.match(r"^feature_(\d+)$", s)
    if n:
        s = n.group(1)
    return s


def postprocess_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process FAISS ChemEmbed results:
        1. Normalise Unique_ID -> feature_id
        2. Keep SMILES columns
        3. Reorder columns to match the original ChemEmbed output style:
           feature_id,
           Top_1_cosine ... Top_N_cosine,
           Top_1_SMILE ... Top_N_SMILE,
           Top_1_InChIKey ... Top_N_InChIKey
    """
    df = raw_df.copy()

    if "Unique_ID" in df.columns:
        df["feature_id"] = df["Unique_ID"].apply(clean_unique_id)
        df = df.drop(columns=["Unique_ID"])
    elif "feature_id" not in df.columns:
        warnings.warn("No Unique_ID or feature_id column — assigning sequential IDs.")
        df["feature_id"] = [f"feature_{i+1:04d}" for i in range(len(df))]

    def _top_number(col_name: str) -> int:
        match = re.search(r"Top_(\d+)_", col_name)
        return int(match.group(1)) if match else 999999

    cosine_cols = sorted(
        [c for c in df.columns if re.match(r"Top_\d+_cosine$", c)],
        key=_top_number,
    )

    smile_cols = sorted(
        [c for c in df.columns if re.match(r"Top_\d+_SMILE$", c)],
        key=_top_number,
    )

    inchikey_cols = sorted(
        [c for c in df.columns if re.match(r"Top_\d+_InChIKey$", c)],
        key=_top_number,
    )

    ordered_cols = ["feature_id"] + cosine_cols + smile_cols + inchikey_cols
    remaining_cols = [c for c in df.columns if c not in ordered_cols]

    return df[ordered_cols + remaining_cols]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("ChemEmbed — FAISS GPU Edition")
    print("=" * 60)
    print(f"Config: {args.config}\n")

    cfg = load_config(args.config)

    ionization    = cfg["ionization"]
    adduct_sign   = "+" if ionization == "pos" else "-"
    adduct_string = "[M+H]+" if ionization == "pos" else "[M-H]-"
    recomp        = cfg["recompute"]

    print(f"Data path  : {cfg['treated_data_path']}")
    print(f"Ionization : {ionization}")
    print(f"ChemEmbed  : {cfg['chemembed_root']}")
    print(f"Top-N      : {cfg['top_n_candidates']}")
    print(f"faiss_k    : {cfg['faiss_k']}")
    print(f"batch_size : {cfg['batch_size']}")
    print(f"Recompute  : {recomp}\n")

    # ── Device selection ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("  (No CUDA GPU detected — running on CPU)\n")

    # ── Load reference DB ONCE before the sample loop ────────────────────────
    _tol_ppm = float(cfg.get("precursor_tolerance_ppm", 0) or 0)
    ref_cache = load_reference_cache(cfg["reference_database"], adduct_sign,
                                     build_tolerance_index=_tol_ppm > 0)

    # ── Sample loop ───────────────────────────────────────────────────────────
    # Support both modes:
    #   1. Batch mode: treated_data_path is a parent folder containing sample dirs.
    #   2. Single-sample mode: treated_data_path points directly to one sample dir.
    #
    # The SLURM array script uses single-sample mode by replacing treated_data_path
    # with SAMPLE_DIR. Therefore, we must detect this case and still save under:
    #   <sample>/<ionization>/chemembed_annotation/
    path = os.path.normpath(cfg["treated_data_path"])
    basename = os.path.basename(path)
    self_metadata = os.path.join(path, basename + "_metadata.tsv")

    if os.path.isfile(self_metadata):
        # treated_data_path already points to one sample directory.
        # Convert it into the same internal representation as batch mode:
        #   path = parent directory
        #   samples_dir = [sample_name]
        samples_dir = [basename]
        path = os.path.dirname(path)
        print(f"[ChemEmbed] Single-sample mode: processing {basename} under {path}")
    else:
        # treated_data_path points to a parent directory containing samples.
        samples_dir = sorted(
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        )
        print(f"[ChemEmbed] Batch mode: found {len(samples_dir)} candidate sample(s) under {path}")

    n_ok = n_fail = n_skip = 0

    for directory in tqdm(samples_dir, desc="ChemEmbed"):
        metadata_path = os.path.join(path, directory, directory + "_metadata.tsv")
        try:
            metadata = pd.read_csv(metadata_path, sep="\t")
        except (FileNotFoundError, NotADirectoryError):
            continue
        except Exception as e:
            print(f"Skipping {directory}: {e}")
            continue

        if "sample_type" not in metadata.columns or metadata.empty:
            continue
        if metadata["sample_type"].iloc[0] != "sample":
            continue

        # Follow original chemembed_by_file.py output/input convention:
        # prefer the preprocessed SIRIUS MGF if present, then raw SIRIUS,
        # then GNPS features MGF.
        # Old layout: files under <sample_dir>/<ionization>/
        sirius_mgf_pp  = os.path.join(path, directory, ionization, f"{directory}_sirius_{ionization}_preprocessed.mgf")
        sirius_mgf_raw = os.path.join(path, directory, ionization, f"{directory}_sirius_{ionization}.mgf")
        gnps_mgf       = os.path.join(path, directory, ionization, f"{directory}_features_ms2_{ionization}.mgf")
        # New flat layout: files directly in <sample_dir>/
        sirius_mgf_pp_flat  = os.path.join(path, directory, f"{directory}_sirius_{ionization}_preprocessed.mgf")
        sirius_mgf_raw_flat = os.path.join(path, directory, f"{directory}_sirius_{ionization}.mgf")
        gnps_mgf_flat       = os.path.join(path, directory, f"{directory}_features_ms2_{ionization}.mgf")

        if os.path.isfile(sirius_mgf_pp) and os.path.getsize(sirius_mgf_pp) > 0:
            mgf_path = sirius_mgf_pp; flat_layout = False
            mgf_source = "SIRIUS preprocessed"
        elif os.path.isfile(sirius_mgf_pp_flat) and os.path.getsize(sirius_mgf_pp_flat) > 0:
            mgf_path = sirius_mgf_pp_flat; flat_layout = True
            mgf_source = "SIRIUS preprocessed (flat)"
        elif os.path.isfile(sirius_mgf_raw) and os.path.getsize(sirius_mgf_raw) > 0:
            mgf_path = sirius_mgf_raw; flat_layout = False
            mgf_source = "SIRIUS raw"
        elif os.path.isfile(sirius_mgf_raw_flat) and os.path.getsize(sirius_mgf_raw_flat) > 0:
            mgf_path = sirius_mgf_raw_flat; flat_layout = True
            mgf_source = "SIRIUS raw (flat)"
        elif os.path.isfile(gnps_mgf) and os.path.getsize(gnps_mgf) > 0:
            mgf_path = gnps_mgf; flat_layout = False
            mgf_source = "GNPS"
        elif os.path.isfile(gnps_mgf_flat) and os.path.getsize(gnps_mgf_flat) > 0:
            mgf_path = gnps_mgf_flat; flat_layout = True
            mgf_source = "GNPS (flat)"
        else:
            continue

        # Canonical layout: annotations/chemembed/ (was chemembed_annotation/).
        if flat_layout:
            output_dir = os.path.join(path, directory, "annotations", "chemembed")
        else:
            output_dir = os.path.join(path, directory, ionization, "annotations", "chemembed")
        results_csv = os.path.join(output_dir, "chemembed_results.csv")

        if not recomp and os.path.isfile(results_csv) and os.path.getsize(results_csv) > 100:
            n_skip += 1
            continue

        print(f"\nSample: {directory}")
        print(f"  MGF source: {mgf_source} ({os.path.basename(mgf_path)})")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Parse MGF
            spectra, id_strategy = parse_mgf(mgf_path)
            if not spectra:
                print("  No spectra in MGF, skipping")
                n_fail += 1
                continue
            print(f"  Parsed {len(spectra)} spectra (ID: {id_strategy})")

            # Convert to MSP
            msp_path = os.path.join(output_dir, "converted_spectra.msp")
            n_written, n_skipped_peaks, n_skipped_adduct = write_msp(
                spectra, msp_path, adduct_string, ionization
            )
            print(f"  MSP: {n_written} written | "
                  f"{n_skipped_peaks} no peaks | {n_skipped_adduct} bad adduct")

            if n_written == 0:
                print("  No compatible spectra, skipping")
                n_fail += 1
                continue

            ce_cfg = {
                "msp_output":          msp_path,
                "preprocessed_data":   os.path.join(output_dir, "preprocessed_data.pkl"),
                "model_path_positive": cfg["model_path_positive"],
                "model_path_negative": cfg["model_path_negative"],
                "tolerance":           cfg["tolerance"],
                "max_mz":              cfg["max_mz"],
                "resolution":          cfg["resolution"],
                "intensity_threshold": cfg["intensity_threshold"],
                "top_n_candidates":    cfg["top_n_candidates"],
                "batch_size":          cfg["batch_size"],
                "num_workers":         cfg["num_workers"],
                "faiss_k":             cfg["faiss_k"],
                "precursor_tolerance_ppm": float(cfg.get("precursor_tolerance_ppm", 0) or 0),
                "adduct":              adduct_sign,
            }

            raw_results = run_chemembed(ce_cfg, ref_cache, device)

            if raw_results is None or len(raw_results) == 0:
                print("  No predictions returned")
                n_fail += 1
                continue

            results_df = postprocess_results(raw_results)
            results_df.to_csv(results_csv, index=False)
            print(f"  Saved: {results_csv} ({len(results_df)} rows)")
            n_ok += 1

        except Exception as e:
            import traceback
            print(f"  Failed: {e}")
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"ChemEmbed complete: {n_ok} OK | {n_fail} failed | {n_skip} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()