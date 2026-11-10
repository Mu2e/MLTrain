// Time the TrkQual models in C++, one track at a time, with the calls TrackQuality_module.cc makes.
//
// usage: trkqual_timing OUTDIR FEATURES NFEATURES REPEATS MODEL...
//
// FEATURES is a float32 file of NFEATURES values per track.  For each MODEL (.onnx or .ubj) it
// loads the model as the module does (ONNXRuntime with default session options; the XGBoost C API
// with nthread 1), scores every track REPEATS times, one track per call, and writes:
//   - OUTDIR/<model file name>.scores.f32: the scores of the last pass, so they can be checked
//     against python's;
//   - one JSON line on stdout: wall-clock and CPU time per track for each pass.
// Run by trkqual_timing.sh in the Musing's environment.

#include "onnxruntime/onnxruntime_cxx_api.h"
#include <xgboost/c_api.h>

#include <cmath>
#include <ctime>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  double cpu_seconds() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
  }

  bool ends_with(const std::string& s, const std::string& end) {
    return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
  }

  void check_xgb(int rc, const char* what) {
    if (rc != 0) throw std::runtime_error(std::string(what) + " failed: " + XGBGetLastError());
  }

  // The ANN as TrackQuality_module.cc runs it
  class Ann {
  public:
    Ann(const std::string& path, size_t n_features) :
      env_(ORT_LOGGING_LEVEL_WARNING, "ONNXInference"),
      session_(env_, path.c_str(), options_),
      input_name_(session_.GetInputNameAllocated(0, allocator_)),
      output_name_(session_.GetOutputNameAllocated(0, allocator_)),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
      shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      size_t total = 1;
      for (auto& dim : shape_) {
        if (dim == -1) dim = 1;
        total *= dim;
      }
      if (total != n_features) {
        throw std::runtime_error(path + " expects " + std::to_string(total) + " features, the file has " +
                                 std::to_string(n_features));
      }
    }
    float score(float* features, size_t n_features) {
      Ort::Value input = Ort::Value::CreateTensor<float>(memory_info_, features, n_features, shape_.data(), shape_.size());
      const char* input_names[] = {input_name_.get()};
      const char* output_names[] = {output_name_.get()};
      auto output = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
      return output[0].GetTensorMutableData<float>()[0];
    }

  private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::AllocatedStringPtr input_name_;
    Ort::AllocatedStringPtr output_name_;
    Ort::MemoryInfo memory_info_;
    std::vector<int64_t> shape_;
  };

  // The BDT as TrackQuality_module.cc runs it
  class Bdt {
  public:
    Bdt(const std::string& path, size_t n_features) {
      check_xgb(XGBoosterCreate(nullptr, 0, &booster_), "XGBoosterCreate");
      check_xgb(XGBoosterLoadModel(booster_, path.c_str()), "XGBoosterLoadModel");
      check_xgb(XGBoosterSetParam(booster_, "nthread", "1"), "XGBoosterSetParam(nthread)");
      bst_ulong n = 0;
      check_xgb(XGBoosterGetNumFeature(booster_, &n), "XGBoosterGetNumFeature");
      if (n != n_features) {
        throw std::runtime_error(path + " expects " + std::to_string(n) + " features, the file has " +
                                 std::to_string(n_features));
      }
    }
    ~Bdt() { XGBoosterFree(booster_); }
    float score(float* features, size_t n_features) {
      DMatrixHandle dmat;
      check_xgb(XGDMatrixCreateFromMat(features, 1, n_features, NAN, &dmat), "XGDMatrixCreateFromMat");
      bst_ulong out_len = 0;
      const float* out = nullptr;
      if (XGBoosterPredict(booster_, dmat, 0, 0, 0, &out_len, &out) != 0) {
        XGDMatrixFree(dmat);
        check_xgb(1, "XGBoosterPredict");
      }
      if (out_len < 1 || out == nullptr) {
        XGDMatrixFree(dmat);
        throw std::runtime_error("XGBoosterPredict returned no result");
      }
      float result = out[0];
      check_xgb(XGDMatrixFree(dmat), "XGDMatrixFree");
      return result;
    }

  private:
    BoosterHandle booster_ = nullptr;
  };

  std::string join(const std::vector<double>& v) {
    std::ostringstream s;
    s.precision(6);
    for (size_t i = 0; i < v.size(); ++i) s << (i ? ", " : "") << v[i];
    return s.str();
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: trkqual_timing OUTDIR FEATURES NFEATURES REPEATS MODEL...\n";
    return 2;
  }
  try {
    const std::string outdir = argv[1];
    const std::string feature_file = argv[2];
    const size_t n_features = std::stoul(argv[3]);
    const int repeats = std::stoi(argv[4]);

    std::ifstream in(feature_file, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + feature_file);
    const size_t bytes = in.tellg();
    if (bytes == 0 || bytes % (n_features * sizeof(float)) != 0) {
      throw std::runtime_error(feature_file + " is not a whole number of " + std::to_string(n_features) + "-feature rows");
    }
    std::vector<float> features(bytes / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(features.data()), bytes);
    const size_t n_tracks = features.size() / n_features;

    for (int m = 5; m < argc; ++m) {
      const std::string path = argv[m];
      std::unique_ptr<Ann> ann;
      std::unique_ptr<Bdt> bdt;
      std::function<float(float*)> score;
      if (ends_with(path, ".onnx")) {
        ann = std::make_unique<Ann>(path, n_features);
        score = [&](float* f) { return ann->score(f, n_features); };
      } else if (ends_with(path, ".ubj")) {
        bdt = std::make_unique<Bdt>(path, n_features);
        score = [&](float* f) { return bdt->score(f, n_features); };
      } else {
        throw std::runtime_error(path + ": expected a .onnx or .ubj model file");
      }

      // one copy of a track's features per call, as the module refills its features vector
      std::vector<float> row(n_features);
      std::vector<float> scores(n_tracks);
      auto run = [&]() {
        for (size_t i = 0; i < n_tracks; ++i) {
          std::copy(features.begin() + i * n_features, features.begin() + (i + 1) * n_features, row.begin());
          scores[i] = score(row.data());
        }
      };
      run();  // warm-up pass, not timed
      std::vector<double> wall_us, cpu_us;
      for (int r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const double c0 = cpu_seconds();
        run();
        const double c1 = cpu_seconds();
        const auto t1 = std::chrono::steady_clock::now();
        wall_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / n_tracks);
        cpu_us.push_back((c1 - c0) * 1e6 / n_tracks);
      }

      const std::string name = path.substr(path.find_last_of('/') + 1);
      const std::string score_file = outdir + "/" + name + ".scores.f32";
      std::ofstream out(score_file, std::ios::binary);
      out.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(float));
      if (!out) throw std::runtime_error("cannot write " + score_file);

      std::cout << "{\"model\": \"" << path << "\", \"tracks\": " << n_tracks
                << ", \"wall_us_per_track\": [" << join(wall_us) << "], \"cpu_us_per_track\": [" << join(cpu_us)
                << "], \"scores\": \"" << score_file << "\"}" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "trkqual_timing: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
