/// \file
/// Test an RBDT model against XGBoost predictions from a CSV file.
///
/// Usage:
///   python3 GenerateTestData.py model.pkl test_data.csv
///   root -l -b 'TestBDTInference.C("test_model.root", "test_data.csv")'

#include "TMVA/RBDT.hxx"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

void TestBDTInference(const char* rootpath, const char* csvpath) {

    // Derive key name from file path (strip directory and .root extension)
    std::string key = std::string(rootpath);
    auto slash = key.rfind('/');
    if (slash != std::string::npos) key = key.substr(slash + 1);
    auto dot = key.rfind(".root");
    if (dot != std::string::npos) key = key.substr(0, dot);

    std::cout << "Loading RBDT from " << rootpath << " with key \"" << key << "\"" << std::endl;
    TMVA::Experimental::RBDT bdt(key, rootpath);

    // Read CSV: columns are f0, f1, ..., fN, xgb_prediction
    std::ifstream csv(csvpath);
    std::string line;
    std::getline(csv, line); // skip header

    int nPassed = 0, nFailed = 0, nTotal = 0;
    double maxDiff = 0;

    while (std::getline(csv, line)) {
        std::stringstream ss(line);
        std::vector<float> features;
        float val;
        char comma;

        // Read all comma-separated values
        std::vector<float> all_values;
        while (ss >> val) {
            all_values.push_back(val);
            ss >> comma;
        }

        // Last column is the XGBoost prediction
        float xgb_pred = all_values.back();
        all_values.pop_back();

        auto rbdt_output = bdt.Compute(all_values);
        float rbdt_pred = rbdt_output[0];
        double diff = std::fabs(rbdt_pred - xgb_pred);

        if (diff > maxDiff) maxDiff = diff;

        if (diff > 1e-5) {
            if (nFailed < 5) { // print first few failures
                std::cout << "MISMATCH sample " << nTotal
                          << ": XGBoost=" << xgb_pred
                          << " RBDT=" << rbdt_pred
                          << " diff=" << diff << std::endl;
            }
            nFailed++;
        } else {
            nPassed++;
        }
        nTotal++;
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total samples: " << nTotal << std::endl;
    std::cout << "Passed (diff < 1e-5): " << nPassed << std::endl;
    std::cout << "Failed: " << nFailed << std::endl;
    std::cout << "Max absolute difference: " << maxDiff << std::endl;

    if (nFailed == 0) {
        std::cout << "\nAll predictions match!" << std::endl;
    }
}
