/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro parses a .onnx file
/// into RModel object and further generating the .hxx header files for inference.
///
/// \macro_code
/// \macro_output
/// \author Sanjiban Sengupta
/// modified by A. Edmonds (2025)

#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"
#include <iostream>

using namespace TMVA::Experimental;

// Create a TMVA::SOFIE model from an ONNX model
void CreateInference(std::string modelname = "model.onnx", std::string infername = "model.hxx"){

  SOFIE::RModelParser_ONNX parser;
  SOFIE::RModel model = parser.Parse(modelname, true);

  //Generating inference code
  model.Generate();

  // Write the code in a file (e.g. model.hxx, model.dat)
  model.OutputGenerated(infername);
}
