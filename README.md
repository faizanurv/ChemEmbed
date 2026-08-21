# A deep learning framework for metabolite identification using enhanced MS/MS data and multidimensional molecular embeddings

A deep learning framework for metabolite identification using enhanced MS/MS data and multidimensional molecular embeddings is designed to process mass spectrometry data, perform predictions using a trained Convolutional Neural Network (CNN) model, and match the predictions against a reference database to identify potential candidate molecules.


## Features

- **Data Preprocessing:** Converts MSP files to structured DataFrames and preprocesses spectra data for model input.
- **Model Prediction:** Utilizes a pre-trained CNN model to predict molecular embeddings from spectra data.
- **Candidate Matching:** Matches predicted embeddings with a reference database to find top candidate molecules based on cosine similarity.
- **Support for Multiple Input Types:** Handles MSP files both **with** and **without** SMILES annotations, controlled via a configuration parameter.
- **Configurable Parameters:** Allows users to adjust parameters like intensity thresholds, resolution, and the number of top candidates via a YAML configuration file.
- **Modular Codebase:** Organized into separate modules for easy maintenance and scalability.

## 🧪 Sample Testing Guide

To test **ChemEmbed**, use the provided Jupyter notebook:

- Open **`ChemEmbed_Sample_Test.ipynb`**.
- Follow the step-by-step instructions included within the notebook.

This notebook demonstrates how to run ChemEmbed on sample data and verify the setup.

## Setup and Configuration

Before running the pipeline, ensure that all dependencies are installed and properly configured. The application is controlled through a `config.yaml` file, which specifies all input/output paths and parameters.

### Installation

```bash
pip install chemembed
```

That installs the code and its Python dependencies. The trained models and the
reference database are distributed separately because of their size — see
**Models and reference database** below.

To install from source instead:

```bash
git clone https://github.com/faizanurv/ChemEmbed.git
cd ChemEmbed
pip install .
```

Note that `pip install chemembed` pulls in PyTorch, which together with its CUDA
libraries occupies roughly 5 GB. On a machine with a small `/tmp` you may need to
redirect pip's scratch space, e.g. `export TMPDIR=/path/with/space`.

### Models and reference database

ChemEmbed needs two model files and one reference database, none of which are
part of the Python package. Download
them from the
[latest release](https://github.com/faizanurv/ChemEmbed/releases/latest):

| File | Size | Purpose |
| --- | --- | --- |
| `model_positive.bin` | 1129.6 MB | CNN weights, positive mode (`[M+H]+`) |
| `model_negative.bin` | 1129.6 MB | CNN weights, negative mode (`[M-H]-`) |
| `chemembed_reference_mol2vec_300d_520k.pkl` | 705.6 MB | 520,083 compounds, 300-dim mol2vec embeddings |
| `chemembed_reference_mol2vec_300d_520k.parquet` | 542.5 MB | the same database as parquet (optional, needs `pyarrow`) |

You need one reference database, not both. The parquet is smaller and safer to
load — `pandas.read_pickle` executes code while unpickling, so a parquet file is
preferable for anything downloaded over the network. It requires `pyarrow`
(`pip install pyarrow`), which the package does not install by default. If you
would rather not add that dependency, use the `.pkl`; both contain identical
data.

```bash
mkdir -p data_model_files && cd data_model_files
BASE=https://github.com/faizanurv/ChemEmbed/releases/download/v1.0.0
curl -LO $BASE/model_positive.bin
curl -LO $BASE/model_negative.bin
curl -LO $BASE/chemembed_reference_mol2vec_300d_520k.parquet   # or the .pkl
```

Verify the downloads before use:

```
b376f329f1b3759e598c9e1fd2e6b252d9f63ed2316baaff3b57be387b0b8c73  model_positive.bin
8165286689322aef9c0486fe470bdafd68162adb9a644d8477810ce8f9fb51ad  model_negative.bin
6ddf348aa6be45db7f5122468b8814c410cbfe87d5b8befc22bc990bc8c561f0  chemembed_reference_mol2vec_300d_520k.pkl
41a88ef1c84e3f0bec2babfb6cac8941ae4fc37168622a94676a3e9c6c753df0  chemembed_reference_mol2vec_300d_520k.parquet
```

```bash
sha256sum -c <<< "b376f329f1b3759e598c9e1fd2e6b252d9f63ed2316baaff3b57be387b0b8c73  model_positive.bin"
```

The reference database is a parquet conversion of `Supplementary File 2.pkl`
from the Zenodo deposit, DOI
[10.5281/zenodo.14778518](https://doi.org/10.5281/zenodo.14778518) (CC-BY-4.0).
Embeddings, SMILES, InChIKeys and precursor masses are unchanged.

**Scope.** The reference database covers 520,083 compounds. Candidate matching
can only return molecules that are in the database, so a compound outside it
cannot be identified regardless of spectral quality, and results obtained with a
differently sized reference are not directly comparable.

### Quick check

```bash
chemembed --input_file_type without_smiles \
          --adduct "+" \
          --msp_file_positive your_spectra.msp \
          --model_path_positive data_model_files/model_positive.bin \
          --reference_database data_model_files/chemembed_reference_mol2vec_300d_520k.parquet \
          --top_n_candidates 5 \
          --prediction_results results.csv
```

Only the flags for the polarity you select are required: with `--adduct "+"` you
need `--msp_file_positive` and `--model_path_positive`, and the negative-mode
equivalents may be omitted.


### How to Run

**Execute the main script using the following command**:

```bash
python main.py --config config.yaml
```

Note: If you do not specify the `--config` argument, the script will default to using `config.yaml`.

## Batch annotation — `chemembed_by_file.py` (FAISS GPU edition)

`main.py` annotates a single MSP file. `chemembed_by_file.py` is the batch entry
point: it walks a directory of samples, reads each sample's GNPS/SIRIUS MGF, and
writes `chemembed_annotation/{converted_spectra.msp, preprocessed_data.pkl,
chemembed_results.csv}` per sample.

```bash
cp chemembed_config.example.yml chemembed_config.yml   # then edit the paths
python chemembed_by_file.py --config chemembed_config.yml
```

`chemembed_config.yml` is gitignored so machine-specific absolute paths are never
committed; `chemembed_config.example.yml` documents every key.

### Where the speed comes from

The batch path produces the same candidates as the original per-sample loop, but
restructures the work:

1. **Reference DB loaded once**, before the sample loop, instead of once per
   sample — and the RDKit `Precursormz` computation over every reference row
   runs once rather than `N_samples` times.
2. **One FAISS index, one search per sample.** All of a sample's query embeddings
   are searched in a single GPU call (~1 ms) instead of a per-query scan of the
   full database.
3. **Precursor m/z post-filter applied to the FAISS results**, preserving the
   original ChemEmbed matching semantics (exact 3-decimal match). A dict-based
   fallback index handles sparse masses whose correct candidates fall outside the
   top `faiss_k` hits, so recall is not traded for speed.
4. **Batched CNN inference on GPU** (`batch_size: 32`) in place of `batch_size=1`
   on CPU — typically 20–50× faster.
5. **`predict_without_smiles` rebuilt** to support `batch_size > 1`, which is what
   makes (4) possible.

`reference_utils.py` also gained a parquet-or-pickle reader and a *self-validating*
`Precursormz` fast path: if the reference already carries a `Precursormz` column,
a random sample of rows is re-derived from SMILES and checked against it for the
requested adduct. Only if the check passes is the stored column trusted; a
negative-mode request, a null-bearing column, or a differently-built database all
fail the check and fall through to the original full recompute. Correctness is
preserved in every case — only the common positive-mode path gets faster.

### Performance keys

| Key | Default | Notes |
| --- | --- | --- |
| `batch_size` | `32` | CNN inference batch size. Raise to 64/128 if the GPU has memory. |
| `num_workers` | `0` | DataLoader worker processes. `4` is reasonable on a multi-core node. |
| `faiss_k` | `200` | FAISS candidates retrieved before the precursor post-filter. Raise if the run reports a high "Fallback used" count. |
| `recompute` | `false` | `true` re-runs samples that already have `chemembed_results.csv`. |

### Requirement

FAISS is required by this entry point and is **not** in `requirements.txt`, since
the correct build depends on your hardware:

```bash
conda install -c pytorch faiss-gpu   # GPU (recommended)
conda install -c pytorch faiss-cpu   # CPU fallback
```

Without a GPU the code still runs — it builds a CPU index and reports
`CPU (no GPU detected)` — but points 2 and 4 lose most of their benefit.

### Per-scan variant — `chemembed_per_spectrum.py`

By default `merge_spectra_by_feature` collapses every MS2 scan of a feature into a
single spectrum. `chemembed_per_spectrum.py` annotates each scan separately,
which is useful when scans differ in collision energy:

```bash
python chemembed_per_spectrum.py \
    --mgf      <sample>_per_spectrum_combined.mgf \
    --manifest <sample>_spectrum_manifest.tsv \
    --config   chemembed_config.yml \
    --out      <sample>_chemembed_per_spectrum_results.tsv
```

It imports the model, reference cache, matching and post-processing from
`chemembed_by_file.py`, so behaviour stays identical, and joins results back to
the manifest to recover `feature_id` / `scan_number` / `collision_energy` /
`dissociation_method`.


### Run as a command line interface

If you use 

```bash
pip install .
```

pip can install the prerequisites and create an executable script which can be run where all the settings of the config file need to be passes as arguments to the executable. 

```bash
chemembed --input_file_type with_smiles \
            --adduct "+" \
            --msp_file_positive MSP_FILE_POSITIVE.msp \
            --msp_file_negative MSP_FILE_NEGATIVE.msp \
            --model_path_positive MODEL_PATH_POSITIVE.bin \
            --model_path_negative MODEL_PATH_NEGATIVE.bin \
            --reference_database REFERENCE_DATABASE.pkl \
            --intensity_threshold 1 \
            --tolerance 0.01 \
            --resolution  0.01 \
            --max_mz  700 \
            --top_n_candidates 7 \
            --preprocessed_data PREPROCESSED_DATA.pkl \
            --prediction_results PREDICTION_RESULTS.csv
```

### Configuration of `config.yaml` File

The application is configured through a `config.yaml` file, which contains several sections:

#### Input Files
- `msp_file`: Path to the input MSP file containing spectra data.
- `reference_database`: Path to the reference database pickle file.
- `model_path`: Path to the pre-trained CNN model file.

#### Output Files
- `preprocessed_data`: Path where the preprocessed data will be saved (pickle format).
- `prediction_results`: Path for the final prediction results CSV file. Supports variable substitution for `top_n_candidates`.

#### Parameters
- `top_n_candidates`: Number of top candidate molecules to retrieve from the reference database. (default: 5)
- `input_file_type`: Type of the input MSP file. Options are 'with_smiles' and 'without_smiles'. (default: 'with_smiles')

#### Example `config.yaml` File

```yaml
Edit the config.yaml file to set up paths and parameters based on your requirements. Key options include:

Input Files:

msp_file_positive: Path to the positive mode spectra file (input_spectra_with_smile.msp if containing SMILES, or input_spectra.msp otherwise).
msp_file_negative: Path to the negative mode spectra file (sample_negative_file_without_smile.msp if not containing SMILES, or sample_negative_file.msp otherwise).
reference_database: Path to the reference database file (e.g., sample_reference_database.pkl).
Model Files:
model_path_positive: Path to the trained positive mode model.
model_path_negative: Path to the trained negative mode model.
Output Files:

preprocessed_data: Path to save preprocessed data.
prediction_results: Path to save prediction results.
Parameters:

top_n_candidates – adjust as needed.
Input Type and Adduct:
input_file_type: Set as with_smiles or without_smiles.
adduct: Choose '-' for M-H or '+' for M+H based on the adduct type.

```

### Using Different Input File Types

- `with_smiles`: Use this option if your MSP file includes SMILES strings for each spectrum. The pipeline will utilize SMILES information during data processing and candidate matching, including Tanimoto similarity calculations.

- `without_smiles`: Use this option if your MSP file does not include SMILES strings. The pipeline will process the data accordingly and perform candidate matching without relying on SMILES information.

### MSP File Format Guide
#### 1. MSP File with SMILES
This format includes SMILES notation, which provides the molecule's structure, followed by metadata and peak data.
##### Format:
```bash smile: <SMILES notation>
smile: <smile >
Precursor: <precursor m/z value>
Adduct: <adduct type>
Num Peaks: <number of peaks>
<m/z> <intensity>
<m/z> <intensity> 
... 
##### Example:
smile: Clc1ccc(cc1)S(=O)(=O)NC2CC2
Precursor: 230.0048
Adduct: [M+H]+
Num Peaks: 16
63.9623 0.15257457101400854
64.9701 0.32720519597920816
...

```

#### 2. MSP File without SMILES

This format does not include SMILES notation and instead begins with a unique identifier followed by metadata and peak data.
##### Format:
```bash
Name: <unique identifier>
Precursor: <precursor m/z value>
Adduct: <adduct type>
Num Peaks: <number of peaks>
<m/z> <intensity>
<m/z> <intensity>
...

##### Example:
Name: ID1
Precursor: 478.1471
Adduct: [M-H]-
Num Peaks: 101
130.9882 11.57
143.6251 2.41
...
```

#### Key Fields in MSP Files
- SMILES (if present): A line representing the molecular structure.
- Precursor: The m/z value of the precursor ion.
- Adduct: Specifies the adduct type (e.g., [M-H]-).
- Num Peaks: Total number of peaks in the entry.
- Peak Data: m/z and intensity values for each peak, with one peak per line.




### Dependencies

Ensure you have the following libraries installed:
- Python 3.6+
- pandas >= 1.0.0
- numpy >= 1.18.0
- rdkit >= 2020.09.1
- torch >= 1.5.0
- scipy >= 1.4.0
- pyyaml >= 5.3.0
- argparse (Built-in)

Note: `rdkit` is best installed via conda due to its dependencies.

### Custom Modules

Ensure the `cnn_train` module is present in your project and contains `up_cnn_model.py` and `spectra_inference_dataset_loader.py`.

### Project Structure

```
spectra2moleculeCombine/
├── data_processing.py                  # Functions related to data preprocessing
├── model_utils.py                      # Functions related to model loading and prediction
├── reference_utils.py                  # Functions related to reference database processing
├── main.py                             # Main script to run the pipeline
├── config.yaml                         # Configuration file
├── data_loaders/                       # Directory containing data loader modules
│   ├── __init__.py
│   ├── inference_dataset_loader.py          # For 'with_smiles' input type
│   └── spectra_inference_dataset_loader.py  # For 'without_smiles' input type
├── cnn_train/                          # Directory containing custom model modules
│   ├── __init__.py
│   └── up_cnn_model.py
├── input_spectra.msp                   # Input spectra file
├── sample_reference_database.pkl       # Reference database file
├── requirements.txt                    # List of required Python packages
└── README.md                           # Project documentation
```

### Additional Information

#### Adjusting the Python Path

If your project relies on locally stored libraries or specific directories that are not installed in standard Python paths, you may need to adjust the Python path. Here's how you can do it within your script:

```python
import sys

# Adjust the path to include the directory where your local libraries are stored
sys.path.append('path/to/your/library')
```

#### Variable Substitution in `config.yaml`

The `prediction_results` filename in `config.yaml` can include `${top_n_candidates}` which will be replaced with the actual number specified in `top_n_candidates`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- RDKit: Open-source cheminformatics software.
- PyTorch: Deep learning framework used for model implementation.

Enjoy using the Mass Spectrometry Data Processing and Prediction Pipeline!

If you have any questions or need assistance, feel free to reach out.
```
