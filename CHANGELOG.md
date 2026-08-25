# Changelog

All notable changes to ChemEmbed are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Release notes on GitHub are generated from the entry for the version being released,
so this file is the single source for what changed.

## [1.2.0] - Unreleased

### Fixed

- **Precursor candidate retrieval now uses a mass tolerance window instead of an exact
  3-decimal bucket.** The previous filter had two independent defects that together
  discarded most correct candidates before scoring:

  - The reference database stores `round(mass, 3)` while the query was reduced by
    string *truncation*, so the equality test failed whenever the 4th decimal digit was
    5 or above. Measured over random masses this missed **49.9%** of compounds *even
    when the query mass was exact* — the compound was present in the database and was
    never retrieved.
  - A 3-decimal bucket is a ±0.0005 Da window, roughly 1.2 ppm at m/z 400, which is
    narrower than the mass accuracy of any real instrument. At a realistic 2 ppm of
    measurement error the miss rate was 61.7%.

  Retrieval now selects every reference within `precursor_tolerance_ppm` of the
  measured precursor. The window is widened by 0.0005 Da to absorb the quantisation
  already present in the stored `Precursormz` column, which at m/z 200 would otherwise
  consume half of a 5 ppm tolerance.

  Measured on 500 MassSpecGym `[M+H]+` spectra, scored against the published
  520,083-compound reference, with every spectrum's true structure confirmed present in
  that reference so any miss is a retrieval failure rather than a coverage limit:

  | | exact 3-dp bucket | ±5 ppm window |
  | --- | --- | --- |
  | Spectra returning any candidate | 317 / 500 (63.4%) | 499 / 500 (99.8%) |
  | Hit@1 | 0.220 | **0.306** |
  | Hit@5 | 0.398 | **0.680** |
  | Hit@20 | 0.488 | **0.884** |

  For 183 of 500 answerable spectra the old filter retrieved nothing at all, and before
  the row-loss fix below those 183 would have been absent from the results file
  entirely — a 317-row CSV for a 500-spectrum run, with no error.

- **Spectra with no candidate are no longer dropped from the results.** They previously
  vanished from the output CSV with no row, no warning and no error, so a run over
  1,000 spectra could return 300 rows and appear to have succeeded. Every input
  spectrum now produces exactly one output row, with `NA` candidates and `NaN` cosines
  when nothing matched, and the run reports how many spectra went unmatched. This
  applies to the batch FAISS path as well.

- `match_predictions_to_reference_with_smiles` no longer raises on an unparseable
  SMILES. It called `CalcMolFormula` on `None`, so one bad structure aborted the run.

- Removed per-spectrum debug output (`way to check adduct`, `Postive more`) and the
  global `pd.options.mode.chained_assignment = None`, which mutated pandas settings for
  the host program.

### Changed

- `precursor_tolerance_ppm` **defaults to 5.0** rather than 0. The ppm window shipped in
  1.1.1 but defaulted to the legacy bucket and was documented nowhere, so no user could
  discover it. Setting it to `0` still selects the exact-bucket behaviour for
  reproducing pre-1.2.0 results.
- Candidate retrieval is backed by a sorted mass index searched with `np.searchsorted`,
  replacing a full scan of the reference database for every spectrum.
- Tanimoto similarity in the `with_smiles` path is computed only for the candidates
  actually reported, not for every row in the bucket.

### Added

- `--precursor_tolerance_ppm` on the command line, and `precursor_tolerance_ppm` in the
  batch configuration template.
- `tests/test_precursor_matching.py` — 30 tests covering the retrieval defects above,
  the window's boundaries, and the guarantee that every input produces an output row.
- A `Precursor matching` workflow running those tests, and a `Release` workflow that
  creates the GitHub release from this changelog when a `Release: <version>` commit
  lands on `main`.

## [1.1.1] - 2026-08-24

### Fixed

- `chemembed.__version__` reported `1.1.0` while the distribution metadata said
  `1.1.1`. The version was written out in three places and the bump missed one; it is
  now read from the installed distribution via `importlib.metadata`, so it cannot
  disagree with `pyproject.toml`.

## [1.1.0] - 2026-08-21

### Changed

- The package moved under `src/chemembed/`. Nine top-level modules became one
  importable package, so `import model_utils` is now
  `from chemembed import model_utils`.
- Data download links are pinned to the `v1.0.1` release rather than
  `releases/latest`, so publishing a code-only release no longer breaks them.

### Added

- Continuous integration: the wheel is built under the declared setuptools floor,
  checked for completeness, then installed and exercised on Python 3.9 through 3.12.
- `CITATION.cff`.

## [1.0.1] - 2026-08-21

### Added

- First public release on PyPI.
- Trained models and the 520,083-compound reference database published as release
  assets.

### Fixed

- `Unique_ID` reached the results CSV as the literal text `('ABC',)` because the
  DataLoader collated the per-item string.
- The CLI required both polarities' files even when only one was in use, so a valid
  single-polarity invocation was rejected.
</content>
