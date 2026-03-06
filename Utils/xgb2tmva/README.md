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
Model base_score: 0.12387519
```

This loads the XGBoost model from `model.pkl` (saved via `joblib.dump`) and produces `my_model.txt`: the text dump that RBDT can read. The model's `base_score` is printed and included in the suggested command (see [baseScore note](#basescore-note) below).

If the model is an `XGBClassifier` trained on a pandas DataFrame, the feature names are printed automatically. Otherwise, features are positional (`f0, f1, ...`) matching the training order.

### 2. Create the RBDT ROOT file

Copy the command printed by step 1:

**Input:**

```bash
root -l -b 'CreateBDTInference.C("my_model.txt", 2, true, 0.12387519)'
```

**Example output:**

```
$ root -l -b 'CreateBDTInference.C("my_model.txt", 2, true, 0.12387519)'
root [0]
Processing CreateBDTInference.C("my_model.txt", 2, true, 0.12387519)...
Detected 10 features from text dump
baseScore (probability): 0.12387519 -> log-odds: -1.95654
Loading XGBoost text dump from: my_model.txt
Saved RBDT to my_model.root with key "my_model"
```

This produces `my_model.root` containing the RBDT object keyed as `my_model`.

The number of features is auto-detected from the text dump. The `baseScore` parameter is the model's `base_score` as a probability; `CreateBDTInference.C` converts it to log-odds internally for RBDT.

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
Model base_score: 0.12387519
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

`CreateBDTInference.C` takes `baseScore` as a probability (matching what XGBoost reports) and converts it to log-odds internally, since RBDT expects a raw additive offset in log-odds space. The default is `0.5`, which maps to `logit(0.5) = 0.0` (no offset), where the logit function is ln(p/(p-1)).

`xgb2tmva.py` prints the model's `base_score` and includes it in the suggested `CreateBDTInference.C` command.

## Files

| File | Description |
|------|-------------|
| `xgb2tmva.py` | Convert XGBoost pickle to text dump |
| `CreateBDTInference.C` | Load text dump into RBDT, save to ROOT file |
| `GenerateTestData.py` | Generate test CSV with XGBoost predictions |
| `TestBDTInference.C` | Validate RBDT against XGBoost predictions |
