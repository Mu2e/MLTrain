"""
Sam Grant 2026

Convert XGBoost to a format compatible with TMVA::Experimental::RBDT.

Feature names are extracted automatically from XGBClassifier models.
Use --feature-names to override or for raw Booster models.

Usage:
    python xgb2tmva.py <input_pkl> <output_name>

Example:
    python xgb2tmva.py xgboost_model.pkl xgboost_model

This produces:
    <output_name>.txt - a text dump for RBDT::LoadText()
"""

import argparse
import os
import joblib
import xgboost as xgb

def xgb2tmva(input_pkl, output_name):
    """Load an XGBoost model from a pickle file and save in TMVA-compatible format."""
    # Load pickle
    model = joblib.load(input_pkl)

    # Get the underlying Booster and feature names
    feature_names = None
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise TypeError(f"Expected an XGBoost model, got {type(model)}")

    # Make output path
    os.makedirs(os.path.dirname(output_name) or ".", exist_ok=True)

    # Text dump for RBDT::LoadText()
    txt_path = output_name + ".txt"
    booster.dump_model(txt_path)
    print(f"Saved text dump to '{txt_path}'")

    # Print feature info
    if feature_names:
        print(f"Feature names: {feature_names}")
    print(f"Number of features: {booster.num_features()}")
    # Print instructions
    print(f"\nTo save as an RBDT ROOT file, run:")
    print(f'  root -l -b \'CreateBDTInference.C("{txt_path}")\'')

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description="Convert XGBoost model to TMVA-compatible format")
    parser.add_argument("input_pkl", help="Path to the pickle file containing the XGBoost model")
    parser.add_argument("output_name", help="Output name (without .txt extension)")
    args = parser.parse_args()
    # Run it
    xgb2tmva(args.input_pkl, args.output_name)
