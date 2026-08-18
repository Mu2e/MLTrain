# TrkPID

## Introduction

TrkPID is a machine learning algorithm is trained to differentiate between electrons and muons using information from both the tracker and the calorimeter.

## Workflow

### Dataset creation
EventNtuple data files are used as input to the training.
The ntuples are skimmed using [make_inputs.py](make_inputs.py) which drops unnecessary data as well as adds branches if any are needed (may require setting up an Analysis musing for EventNtuple libraries)

### Model training
The python code provided in the file [TrackPIDTrain.py](TrackPIDTrain.py) is used to:
* skim track information from the input ntuples and store the data in local csv files
* define the neural network architecture
* train the algorithm and save the model weights into an ONNX file named "TrackPID.onnx"
* test the algorithm, providing performance metrics
* generate plots to provide more information on the dataset, how the training went, and how the model perform

Once the model is trained and the weights are saved in an ONNX file, this file can be used by TMVA:SOFIE to generate the inference code
that can be used in Offline.
See `scripts/CreateInference.C` to create this model.

## Version history

### v0
This version has been trained using MDC2020au datasets, and tested on MDC2020au and MDC2020aw, generated using Offline v11_00_00 and EventNtuple v06_07_00.
It was trained using mono-energetic conversion electrons as signal and cosmic ray muons as background.
The input features are:
* E(cluster) - P(track)
* R(cluster)
* p(track) dot x(cluster)
* t(track) - t(cluster)

### v1 (current version)
This version has been trained using MDC2025 datasets:
* signal: nts.mu2e.FlateMinusOnSpill-reco-ntuple.MDC2025-004.root
* background: nts.mu2e.FlatMuMinusOnSpill-reco-ntuple.MDC2025-004.root
It was trained using electrons and muons generated with a flat momentum spectrum from muon stop vertices in the target.
The input features are:
* E(cluster)/P(track)
* t(track) - t(cluster)
* p(chi^2) of the track
* straw hit time vs track time at the hit slope

[compare_hit_slopes.py](compare_hit_slopes.py) compares different methods of using the straw hit time vs. track hit time prediction.
