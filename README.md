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

**Clone this repository to your local machine**:

```bash
git clone https://github.com/massspecdl/ChemEmbed.git
cd ChemEmbed
```

**Create a virtual environment (optional but recommended)**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Install required dependencies**:

```bash
pip install -r requirements.txt
```

**Download data and model from this link**:

```bash
pip install gdown
gdown https://drive.google.com/drive/folders/1GEAiTPPTUsLJxYYOr2zVuAm_74LyAwAw?usp=sharing --folder
unzip Data_and_model_folder.zip
Put all files in folder where you put all codes
```

Note: If a `requirements.txt` file is not present, install the dependencies individually (see Dependencies).

### How to Run

**Execute the main script using the following command**:

```bash
python main.py --config config.yaml
```

Note: If you do not specify the `--config` argument, the script will default to using `config.yaml`.


### Run as a command line interface

If you use 

```bash
pip install -r .
```

pip can install the prerequisites and create an executable script which can be run where all the settings of the config file need to be passes as arguments to the executable. 

```bash
ChemEmbed --input_file_type with_smiles 
            --adduct + 
            --msp_file_positive MSP_FILE_POSITIVE.msp 
            --msp_file_negative MSP_FILE_NEGATIVE.msp 
            --model_path_positive MODEL_PATH_POSITIVE.bin 
            --model_path_negative MODEL_PATH_NEGATIVE.bin 
            --reference_database REFERENCE_DATABASE.pkl
            --intensity_threshold 1 
            --tolerance 0.01 
            --resolution  0.01 
            --max_mz  700 
            --top_n_candidates 7
            --preprocessed_data PREPROCESSED_DATA.pkl 
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
