# cli.py

import argparse
from chemembed_single_file import run


def main():
    """
        A cli targeted invocation, where all the configuration settings come from command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process and predict mass spectrometry data.')
    # input type
    parser.add_argument('--input_file_type', type=str, default='with_smiles', choices=["with_smiles", "without_smiles"], help='Does the input file time include smiles.')
    parser.add_argument('--adduct', type=str, default='+', choices=["+", "-"], help='Choose "-" for M-H or "+" for M+H based on the adduct type.')

    # input files
    parser.add_argument('--msp_file_positive', type=str, help='Path to the positive mode spectra file.')
    parser.add_argument('--msp_file_negative', type=str, help='Path to the negative mode spectra file.')

    # models
    parser.add_argument('--model_path_positive', type=str, help='Path to the trained positive mode model.')
    parser.add_argument('--model_path_negative', type=str, help='Path to the trained negative mode model.')
    parser.add_argument('--reference_database', type=str, required=True, help='Path to your reference database as pkl file.')

    # parameters
    parser.add_argument('--intensity_threshold', type=float, default =1, help='Intensity threshold in percentage.')
    parser.add_argument('--tolerance', type=float, default =0.01, help='Tolerance.')
    parser.add_argument('--resolution', type=float, default =0.01, help='Resolution.')
    parser.add_argument('--max_mz', type=float, default =700, help='Max MZ.')
    parser.add_argument('--top_n_candidates', type=int, default =5, help='Top N candidates.')

    # outputs
    parser.add_argument('--preprocessed_data', type=str, default='preprocessed_data.pkl', help='Path to save preprocessed data as pkl file.')
    parser.add_argument('--prediction_results', type=str, default='prediction_results.csv', help='Path to save prediction results as csv file.')

    args = parser.parse_args()

    if args.adduct == "+" and not args.msp_file_positive:
        parser.error("--msp_file_positive is required when --adduct is '+'")

    if args.adduct == "-" and not args.msp_file_negative:
        parser.error("--msp_file_negative is required when --adduct is '-'")

    if args.adduct == "+" and not args.model_path_positive:
        parser.error("--model_path_positive is required when --adduct is '+'")

    if args.adduct == "-" and not args.model_path_negative:
        parser.error("--model_path_negative is required when --adduct is '-'")


    run(vars(args))


if __name__ == "__main__":
    main()
