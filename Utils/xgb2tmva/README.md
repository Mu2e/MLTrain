# XGBoost to TMVA conversion

Convert an XGBoost model (pickle) to `TMVA::Experimental::RBDT` for use in Offline.

This avoids the XGBoost -> ONNX -> TMVA::SOFIE route, which has compatibility issues. Instead, use XGBoost's native text dump format, which `RBDT::LoadText()` reads directly.

## Setup

**Input:**

```bash
mu2einit
pyenv rootana  # ROOT 6.32 with RBDT support
```

**Example output:**
```
$ mu2einit
$ pyenv rootana
Activating Mu2e Python environment: rootana current
Run 'deactivate' to exit the environment
```

## Pipeline

### 1. Export the text dump

**Input:**

```bash
python xgb2tmva.py model.pkl my_model
```

**Example output:**

```
$ python xgb2tmva.py model.pkl my_model
Saved text dump to 'my_model.txt'
Number of features: 10

To save as an RBDT ROOT file, run:
  root -l -b 'CreateBDTInference.C("my_model.txt")'
```

This loads the XGBoost model from `model.pkl` (saved via `joblib.dump`) and produces `my_model.txt`: the text dump that RBDT can read.

If the model is an `XGBClassifier` trained on a pandas DataFrame, the feature names are supposed to be printed automatically. Otherwise, features are positional (`f0, f1, ...`) matching the training order.

### 2. Create the RBDT ROOT file

**Input:**

```bash
root -l -b 'CreateBDTInference.C("my_model.txt")'
```

**Example output:**

```
$ root -l -b 'CreateBDTInference.C("my_model.txt")'
root [0] 
Processing CreateBDTInference.C("my_model.txt")...
Detected 10 features from text dump
Loading XGBoost text dump from: my_model.txt
Saved RBDT to my_model.root with key "my_model"

To use in C++:
  TMVA::Experimental::RBDT bdt("my_model", "my_model.root");
  std::vector<float> input = { f0, f1, ..., f9 }; // 10 features in training order
  auto output = bdt.Compute(input);  // output[0] is the classifier score
```

This produces `my_model.root` containing the RBDT object keyed as `my_model`.

The number of features is auto-detected from the text dump. The `baseScore` defaults to `0.0`, which is correct for XGBoost >= 2.0 (see [baseScore note](#basescore-note) below).

### 3. Use in C++

```cpp
TMVA::Experimental::RBDT bdt("my_model", "my_model.root");
std::vector<float> input = { f0, f1, ..., fN };  // features in training order
auto output = bdt.Compute(input);  // output[0] is the classifier score
```

The output score is compatible with `MVAResultInfo` in EventNtuple.

## Validation

To verify the conversion produces identical predictions:

**Input:**
```bash
# Generate test data with XGBoost predictions
python GenerateTestData.py model.pkl test_data.csv

# Compare against RBDT predictions
root -l -b 'TestBDTInference.C("my_model.root", "test_data.csv")'
```

**Example output:**
```
$ python GenerateTestData.py model.pkl test_data.csv
Saved 100 test samples with 10 features to test_data.csv
Prediction range: [0.000001, 0.001771]
Model base_score: 0.5
(Pass this to CreateBDTInference.C if it differs from 0.5!)

$ root -l -b 'TestBDTInference.C("my_model.root", "test_data.csv")'
root [0] 
Processing TestBDTInference.C("my_model.root", "test_data.csv")...
Loading RBDT from my_model.root with key "my_model"

=== Results ===
Total samples: 100
Passed (diff < 1e-5): 100
Failed: 0
Max absolute difference: 0

All predictions match
```

All samples should pass with `diff < 1e-5`.

### baseScore note

XGBoost >= 2.0 bakes `base_score` into the tree leaf values, so `CreateBDTInference.C` defaults to `baseScore = 0.0`. If you see a systematic offset in predictions, try `baseScore = 0.5` for older models. `GenerateTestData.py` prints the model's `base_score` to help diagnose this.

## Files

| File | Description |
|------|-------------|
| `xgb2tmva.py` | Convert XGBoost pickle to text dump |
| `CreateBDTInference.C` | Load text dump into RBDT, save to ROOT file |
| `GenerateTestData.py` | Generate test CSV with XGBoost predictions |
| `TestBDTInference.C` | Validate RBDT against XGBoost predictions |
