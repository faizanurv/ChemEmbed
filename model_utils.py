# model_utils.py

import torch
import pandas as pd

# Assuming you have your model class defined somewhere, e.g., up_cnn_model.py
from cnn_train import up_cnn_model

def load_model(model_path):
    """
    Load the trained model.
    """
    model_cnn = up_cnn_model.CNN_Class()
    model_cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model_cnn.eval()
    return model_cnn


def _unbatch(value):
    """Undo DataLoader's collation of a per-item string.

    Dataset.__getitem__ returns Unique_ID (and SMILES) as plain strings, but
    DataLoader collates every field of a batch, turning "ABC" into ["ABC"].
    With batch_size=1 that one-element sequence was being written straight into
    the results DataFrame, so the output CSV carried the literal text
    "('ABC',)" instead of "ABC" and could not be joined on without stripping.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value

def predict_with_smiles(model, test_loader, input_type):
    """
    Make predictions using the model for 'with_smiles' input.
    """

    df = pd.DataFrame(columns=('Precursor', 'tree_out', 'smile', 'len_frag', 'Unique_ID'))
    count = 0
    with torch.no_grad():
        for inputs in test_loader:
            inputs01 = inputs[0]
            precursor = inputs[1].item()
            smile = _unbatch(inputs[2])
            len_frag = inputs[3].item()
            unique_id = _unbatch(inputs[4])
            outputs1 = model(inputs01)
            tree_out = outputs1.detach().cpu().numpy()
            df.loc[count] = [precursor, tree_out, smile, len_frag, unique_id]
            count += 1
    return df


def predict_without_smiles(model, test_loader, input_type):
    """
    Make predictions using the model for 'without_smiles' input.
    """
    df = pd.DataFrame(columns=('Precursor', 'tree_out', 'len_frag', 'Unique_ID'))
    count = 0
    with torch.no_grad():
        for inputs in test_loader:
            inputs01 = inputs[0]
            precursor = inputs[1].item()  # Extract scalar from tensor
            len_frag = inputs[2].item()   # Extract scalar from tensor
            unique_id = _unbatch(inputs[3])
            outputs1 = model(inputs01)
            tree_out = outputs1.detach().cpu().numpy()
            df.loc[count] = [precursor, tree_out, len_frag, unique_id]
            count += 1
    return df
