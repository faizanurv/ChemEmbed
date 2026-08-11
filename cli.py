# main.py

import pandas as pd
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Import functions from our modules
from data_processing import (
    msp_to_dataframe_with_smiles,
    msp_to_dataframe_without_smiles,
    preprocess_spectra_with_smiles,
    preprocess_spectra_without_smiles,
    process_data_with_smiles,
    process_data_without_smiles
)

from model_utils import (
    load_model,
    predict_with_smiles,
    predict_without_smiles
)

from reference_utils import (
    load_reference_database_with_smiles,
    load_reference_database_without_smiles,
    match_predictions_to_reference_with_smiles,
    match_predictions_to_reference_without_smiles
)


def main():
    """
        A cli targeted invocation, where all the configuration settings come from command line arguments.
    """
    parser = argparse.ArgumentParser(description='Process and predict mass spectrometry data.')
    # input type
    parser.add_argument('--input_file_type', type=str, default='with_smiles', choices=["with_smiles", "without_smiles"], help='Does the input file time include smiles.')
    parser.add_argument('--adduct', type=str, default='+', choices=["+", "-"], help='Choose "-" for M-H or "+" for M+H based on the adduct type.')

    # input files
    parser.add_argument('--msp_file_positive', type=str, required=True, help='Path to the positive mode spectra file.')
    parser.add_argument('--msp_file_negative', type=str, required=True, help='Path to the negative mode spectra file.')

    # models
    parser.add_argument('--model_path_positive', type=str, required=True, help='Path to the trained positive mode model.')
    parser.add_argument('--model_path_negative', type=str, required=True, help='Path to the trained negative mode model.')
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

    input_type = args.input_file_type
    adduct = args.adduct

    if input_type == 'with_smiles':
        # Processing for 'with_smiles'
        if args.adduct == "+":
            print("Positive Point")
            msp_df = msp_to_dataframe_with_smiles(args.msp_file_positive)
        else:
            print("Negative Point")
            msp_df = msp_to_dataframe_with_smiles(args.msp_file_negative)
        norm_df = preprocess_spectra_with_smiles(msp_df, args.intensity_threshold)
        final_up = process_data_with_smiles(norm_df, args.tolerance, args.resolution, args.max_mz)
        final_up.to_pickle(args.preprocessed_data)
        from data_loaders import inference_dataset_loader as data_loader_module
        test_dataset = data_loader_module.class_ls(args.preprocessed_data)
        predict_fn = predict_with_smiles
        reference_loader_fn = load_reference_database_with_smiles
        matcher_fn = match_predictions_to_reference_with_smiles
    else:
        # Processing for 'without_smiles'
        if args.adduct == "+":
            msp_df = msp_to_dataframe_without_smiles(args.msp_file_positive)
        else:
            msp_df = msp_to_dataframe_without_smiles(args.msp_file_negative)
        norm_df = preprocess_spectra_without_smiles(msp_df, args.intensity_threshold)
        final_up = process_data_without_smiles(norm_df, args.tolerance, args.resolution, args.max_mz)
        final_up.to_pickle(args.preprocessed_data)
        from data_loaders import spectra_inference_dataset_loader as data_loader_module
        test_dataset = data_loader_module.class_ls(args.preprocessed_data)
        predict_fn = predict_without_smiles
        reference_loader_fn = load_reference_database_without_smiles
        matcher_fn = match_predictions_to_reference_without_smiles

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             drop_last=True,
                             shuffle=False,
                             num_workers=0)
    if args.adduct == "+":
        #print("Positive model load")
        model_cnn = load_model(args.model_path_positive)
    else:
        #print("Negative model load")
        model_cnn = load_model(args.model_path_negative)

    # Make predictions
    prediction_df = predict_fn(model_cnn, test_loader, input_type)

    # Load and preprocess reference database
    reference_df = reference_loader_fn(args.reference_database, adduct)

    # Match predictions to reference database
    final_results_df = matcher_fn(prediction_df, reference_df, args.top_n_candidates, input_type, adduct)

    # Save final results
    final_results_df.to_csv(args.prediction_results, index=False)

if __name__ == "__main__":
    main()
