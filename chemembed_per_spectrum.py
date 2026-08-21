#!/usr/bin/env python3
"""
chemembed_per_spectrum.py — run ChemEmbed PER SCAN (not merged per feature).

Mirrors the SIRIUS per-spectrum logic: instead of ChemEmbed's default behaviour
(merge_spectra_by_feature collapses every scan of a feature into one spectrum),
this runs ChemEmbed on the combined per-scan MGF produced by
SIRIUS/src/mgf_split_spectra.py, where each MS2 scan is a separate "compound"
identified by feature{ID}_scan{N}.

Because each scan carries a UNIQUE feature{ID}_scan{N} id, ChemEmbed's
feature-level merge degenerates to one spectrum per scan — so each scan gets its
own embedding, FAISS match and top-k candidates.

The heavy ChemEmbed machinery (model, FAISS reference cache, matching,
post-processing) is imported from chemembed_by_file.py so behaviour stays
identical (multi-adduct handling, up_inchikey, etc.).

Output: one TSV with per-scan candidates, joined to the spectrum manifest to
recover feature_id / scan_number / collision_energy / dissociation_method.
Each scan row carries the wide ChemEmbed columns
(Top_1..N_cosine / _SMILE / _InChIKey).

Usage:
    python chemembed_per_spectrum.py \
        --mgf       <sample>_per_spectrum_combined.mgf \
        --manifest  <sample>_spectrum_manifest.tsv \
        --config    ChemEmbed/chemembed_config.yml \
        --out       <sample>_chemembed_per_spectrum_results.tsv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

import torch  # noqa: E402

from chemembed.chemembed_by_file import (  # noqa: E402
    load_config,
    load_reference_cache,
    parse_mgf,
    write_msp,
    run_chemembed,
    postprocess_results,
)


META_COLS = ["spectrum_id", "feature_id", "scan_number",
             "collision_energy", "dissociation_method"]


def run(mgf_path, manifest_path, config_path, out_path):
    mgf_path      = Path(mgf_path)
    manifest_path = Path(manifest_path) if manifest_path else None
    out_path      = Path(out_path)

    if not mgf_path.exists():
        print(f"ERROR: combined MGF not found: {mgf_path}", file=sys.stderr)
        sys.exit(1)

    # load_config also inserts chemembed_root onto sys.path (needed for
    # reference_utils / model imports inside run_chemembed).
    cfg = load_config(config_path)
    ionization    = cfg["ionization"]
    adduct_sign   = "+" if ionization == "pos" else "-"
    adduct_string = "[M+H]+" if ionization == "pos" else "[M-H]-"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chemembed-per-scan] device={device}  ionization={ionization}")

    # Reference FAISS cache (loaded once)
    ref_cache = load_reference_cache(cfg["reference_database"], adduct_sign)

    # ── Parse the combined per-scan MGF (MS2 only; each id = feature{ID}_scan{N}) ──
    spectra, id_strategy = parse_mgf(str(mgf_path))
    if not spectra:
        print("ERROR: no MS2 spectra parsed from combined MGF", file=sys.stderr)
        sys.exit(1)
    print(f"[chemembed-per-scan] parsed {len(spectra)} per-scan spectra "
          f"(ID: {id_strategy})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_path.parent

    # ── MSP (write_msp groups by feature_id; unique per-scan ids → 1 each) ───────
    msp_path = str(work_dir / "per_spectrum_converted.msp")
    n_written, n_no_peaks, n_bad_adduct = write_msp(
        spectra, msp_path, adduct_string, ionization
    )
    print(f"[chemembed-per-scan] MSP: {n_written} scans written | "
          f"{n_no_peaks} no peaks | {n_bad_adduct} incompatible adduct")
    if n_written == 0:
        print("ERROR: no ChemEmbed-compatible scans", file=sys.stderr)
        sys.exit(1)

    # ── Run ChemEmbed embedding + FAISS matching ────────────────────────────────
    ce_cfg = {
        "msp_output":          msp_path,
        "preprocessed_data":   str(work_dir / "per_spectrum_preprocessed.pkl"),
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
        "adduct":              adduct_sign,
    }
    raw_results = run_chemembed(ce_cfg, ref_cache, device)
    if raw_results is None or len(raw_results) == 0:
        print("ERROR: ChemEmbed returned no predictions", file=sys.stderr)
        sys.exit(1)

    results_df = postprocess_results(raw_results)
    # postprocess names the id column "feature_id"; here it holds the per-scan
    # combined id (feature{ID}_scan{N}). Rename to spectrum_id for clarity.
    results_df = results_df.rename(columns={"feature_id": "spectrum_id"})
    results_df["spectrum_id"] = results_df["spectrum_id"].astype(str)

    # ── Join scan metadata from the manifest (parallels SIRIUS collect) ──────────
    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
        manifest["spectrum_id"] = manifest["spectrum_id"].astype(str)
        keep = [c for c in META_COLS if c in manifest.columns]
        merged = manifest[keep].merge(results_df, on="spectrum_id", how="left")
        # status flag for scans with no ChemEmbed candidate
        cosine_cols = [c for c in merged.columns if c.endswith("_cosine")]
        if cosine_cols:
            has_hit = merged[cosine_cols].notna().any(axis=1)
            merged["chemembed_status"] = has_hit.map({True: "ok", False: "no_result"})
        else:
            merged["chemembed_status"] = "no_result"
    else:
        # No manifest: derive feature/scan from the id string as a fallback.
        merged = results_df.copy()
        merged["feature_id"]  = merged["spectrum_id"].str.replace(
            r"^feature(.*)_scan.*$", r"\1", regex=True)
        merged["scan_number"] = merged["spectrum_id"].str.replace(
            r"^feature.*_scan(.*)$", r"\1", regex=True)
        merged["chemembed_status"] = "ok"
        front = ["spectrum_id", "feature_id", "scan_number"]
        merged = merged[front + [c for c in merged.columns if c not in front]]

    merged.to_csv(out_path, sep="\t", index=False)

    n_scans = merged["spectrum_id"].nunique()
    n_hit   = merged.loc[merged.get("chemembed_status", "") == "ok",
                         "spectrum_id"].nunique()
    print(f"[chemembed-per-scan] scans with candidates : {n_hit}/{n_scans}")
    print(f"[chemembed-per-scan] rows written          : {len(merged)}")
    print(f"[chemembed-per-scan] output                : {out_path}")


def main():
    _default_cfg = _HERE / "chemembed_config.yml"
    ap = argparse.ArgumentParser(
        description="Run ChemEmbed per scan on a combined feature{ID}_scan{N} MGF."
    )
    ap.add_argument("--mgf",      required=True, type=Path,
                    help="Combined per-scan MGF (from mgf_split_spectra.py).")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Spectrum manifest TSV (spectrum_id, feature_id, scan, CE, method).")
    ap.add_argument("--config",   type=Path, default=_default_cfg,
                    help=f"ChemEmbed config YAML. Default: {_default_cfg}")
    ap.add_argument("--out",      required=True, type=Path,
                    help="Output per-scan results TSV.")
    args = ap.parse_args()
    run(args.mgf, args.manifest, args.config, args.out)


if __name__ == "__main__":
    main()
