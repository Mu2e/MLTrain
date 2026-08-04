# TrkOnlyPID

## Introduction

TrkOnlyPID is a machine learning algorithm is trained to differentiate between electrons and muons using information only from the tracker.

## Workflow

### Dataset creation
EventNtuple data files are used as input to the training.
The ntuples are skimmed using [make_inputs.py](make_inputs.py) which drops unnecessary data as well as adds branches if any are needed.

### Model training
The python code provided in the file [TrkOnlyPIDTrain.py](TrkOnlyPIDTrain.py) is used to:
* skim track information from the input ntuples and store the data in local csv files
* define the neural network architecture
* train the algorithm and save the model weights into an ONNX file named "TrkOnlyPID.onnx"
* test the algorithm, providing performance metrics
* generate plots to provide more information on the dataset, how the training went, and how the model perform

Once the model is trained and the weights are saved in an ONNX file, this file can be used by TMVA:SOFIE to generate the inference code
that can be used in Offline (for details about this process, check [this documentation](https://github.com/Mu2e/MLTrain/blob/main/TrkQual/README.md#converting-a-model-for-use-in-offline)).

## Version history

### v0 (current version)
This version has been trained using MDC2025 datasets:
* signal: nts.mu2e.FlateMinusOnSpill-reco-ntuple.MDC2025-004.root
* background: nts.mu2e.FlatMuMinusOnSpill-reco-ntuple.MDC2025-004.root
It was trained using electrons and muons generated with a flat momentum spectrum from muon stop vertices in the target.
The input features are:
* N(active hits) / N(hits)
* N(null ambiguity hits) / N(active hits)
* p(chi^2) of the track
* straw hit time vs track time at the hit slope
