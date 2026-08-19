# Perturbation Analysis Study

This directory contains a suite of notebooks dedicated to evaluating an independent XGBoost gradient-boosted decision tree (BDT) classifier for the Mu2e tracking quality verification pipeline. The focus is to analyze algorithmic stability and domain adaptation under various systematic track features and data distortions.

## File Walkthrough
* **00_TrkQualTrain_main.ipynb:** It was only added to help viewer see how trained BDT model was saved.
* **01_data_perturbation.ipynb:** Responsible for creating and perturbing all of the required data sets.
* **02_factive_nactive_analysis.ipynb:** Investigates the physical properties of track hits, evaluates feature correlations, and isolates the impact of active hit fractions on background rejection.
* **03_momerr_robustness_analysis.ipynb:** A targeted stress-test notebook that subjects the model to severe ±10% and extreme ±50% momentum error variations to evaluate algorithmic stability.
* **04_all_data_perturbation_analysis.ipynb:** The global analysis file. It compiles all perturbed data streams into a unified evaluation framework to compare static cut strategies against adaptive dynamic domain adaptation.

## Methodological Notes on Feature Sensitivity

### Single-Feature Perturbation Logic (`nactive` vs. `factive`)
In Notebook `02`, `nactive` was deliberately perturbed independently as there are multiple features that depend on it, so perturbing it gave an initial understanding and helped to move towards perturbing the `momerr` feature.