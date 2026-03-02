"""
Sam Grant 2026
Generate test data for validating xgboost -> RBDT conversion.
Saves random inputs and their xgb predictions to a csv file,
which can then be compared against RBDT output using TestBDTInference.C.

Usage:
    python GenerateTestData.py <model_pkl> <output_csv> [--n-samples 100]
"""

import argparse
import numpy as np
import joblib
import xgboost as xgb
import json

def generate_test_data(model_pkl, output_csv, n_samples=100):
    model = joblib.load(model_pkl)

    # Get the underlying Booster
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise TypeError(f"Expected an XGBoost model, got {type(model)}")
    n_features = booster.num_features()

    # Generate random inputs 
    np.random.seed(42)
    inputs = np.random.rand(n_samples, n_features).astype(np.float32)

    # Get predictions
    dmatrix = xgb.DMatrix(inputs)
    predictions = booster.predict(dmatrix)

    # Save to CSV
    header = ",".join([f"f{i}" for i in range(n_features)] + ["xgb_prediction"])
    data = np.column_stack([inputs, predictions])
    np.savetxt(output_csv, data, delimiter=",", header=header, comments="")

    print(f"Saved {n_samples} test samples with {n_features} features to {output_csv}")
    print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")

    # Print base_score for use with CreateBDTInference.C
    config = json.loads(booster.save_config())
    base_score = config["learner"]["learner_model_param"]["base_score"]
    print(f"Model base_score: {float(base_score)}")
    print(f"(Pass this to CreateBDTInference.C if it differs from 0.5!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data for RBDT validation")
    parser.add_argument("model_pkl", help="Path to the XGBoost model pickle")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of test samples")
    args = parser.parse_args()

    generate_test_data(args.model_pkl, args.output_csv, args.n_samples)
