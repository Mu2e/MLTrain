/*
    Sam Grant 2026

    Follows from xgb2tmva.py: 
        - Get the text dump output;
        - Load it into TMVA::Experimental::RBDT;
        - Save it to a ROOT file.

    Usage:
        root -l -b 'CreateBDTInference.C("test_model.txt")'

    This produces:
        <basename>.root containing the RBDT object keyed as <basename>
*/

#include "TMVA/RBDT.hxx"
#include "TFile.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// baseScore: XGBoost's base_score as a probability (from GenerateTestData.py).
// RBDT expects an additive offset in log-odds space, so we convert via logit().
// Default 0.5 → logit(0.5) = 0.0 (no offset), which is correct when the
// base_score is already baked into the leaf values (XGBoost >= 2.0).
void CreateBDTInference(const char* txtpath, int nClasses = 2,
    bool logistic = true, float baseScore = 0.5) {

    // Derive output name from input path: strip directory and .txt extension
    std::string outname = std::string(txtpath);
    auto slash = outname.rfind('/');
    if (slash != std::string::npos) outname = outname.substr(slash + 1);
    auto ext = outname.rfind(".txt");
    if (ext != std::string::npos) outname = outname.substr(0, ext);

    // Scan the text dump to find the highest feature index
    int maxIdx = -1;
    {
        std::ifstream in(txtpath);
        std::string line;
        std::regex re("\\[f(\\d+)<");
        std::smatch m;
        while (std::getline(in, line)) {
            auto it = line.cbegin();
            while (std::regex_search(it, line.cend(), m, re)) {
                int idx = std::stoi(m[1].str());
                if (idx > maxIdx) maxIdx = idx;
                it = m.suffix().first;
            }
        }
    }
    int nFeatures = maxIdx + 1;
    std::cout << "Detected " << nFeatures << " features from text dump" << std::endl;

    // Generate generic feature names to match XGBoost's text dump indices
    std::vector<std::string> features;
    for (int i = 0; i < nFeatures; ++i) {
        features.push_back("f" + std::to_string(i));
    }

    // Convert baseScore from probability to log-odds for RBDT
    float baseScoreLogOdds = std::log(baseScore / (1.0f - baseScore));
    std::cout << "Loading XGBoost text dump from: " << txtpath << std::endl;
    std::cout << "baseScore (probability): " << baseScore
              << " -> log-odds: " << baseScoreLogOdds << std::endl;
    auto bdt = TMVA::Experimental::RBDT::LoadText(
        std::string(txtpath), features, nClasses, logistic, baseScoreLogOdds
    );

    std::string rootfile = std::string(outname) + ".root";
    TFile f(rootfile.c_str(), "RECREATE");
    f.WriteObject(&bdt, outname.c_str());
    f.Close();

    std::cout << "Saved RBDT to " << rootfile << " with key \"" << outname << "\"" << std::endl;
    std::cout << "\nTo use in C++:" << std::endl;
    std::cout << "  TMVA::Experimental::RBDT bdt(\"" << outname << "\", \"" << rootfile << "\");" << std::endl;
    std::cout << "  std::vector<float> input = { f0, f1, ..., f" << nFeatures - 1 << " }; // " << nFeatures << " features in training order" << std::endl;
    std::cout << "  auto output = bdt.Compute(input);  // output[0] is the classifier score" << std::endl;
    std::cout << "\nTo validate against XGBoost predictions:" << std::endl;
    std::cout << "  python GenerateTestData.py model.pkl test_data.csv" << std::endl;
    std::cout << "  root -l -b 'TestBDTInference.C(\"" << rootfile << "\", \"test_data.csv\")'" << std::endl;
}
