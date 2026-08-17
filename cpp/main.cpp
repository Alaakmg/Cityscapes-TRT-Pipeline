// Minimal TensorRT C++ inference wrapper.
//
// Loads a serialized engine, preprocesses an image with OpenCV using the same
// ImageNet normalization as training, runs inference, writes a colorized mask.
//
//   ./segdeploy_infer model_fp16.engine input.png output.png
//
// Build on the target device (TensorRT + OpenCV from JetPack):
//   cmake -B build && cmake --build build

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

// RAII wrapper for device memory.
struct DeviceBuffer {
  void* ptr = nullptr;
  explicit DeviceBuffer(size_t bytes) { cudaMalloc(&ptr, bytes); }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
const unsigned char kPalette[8][3] = {
    {0, 0, 0},       {128, 64, 128}, {70, 70, 70}, {220, 220, 0},
    {107, 142, 35},  {70, 130, 180}, {220, 20, 60}, {0, 0, 142}};

std::vector<char> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> buf(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <engine> <input.png> <output.png>\n";
    return 1;
  }

  Logger logger;
  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  const auto plan = readFile(argv[1]);
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine) { std::cerr << "engine deserialization failed\n"; return 1; }
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

  const char* inName = engine->getIOTensorName(0);
  const char* outName = engine->getIOTensorName(1);
  const auto inDims = engine->getTensorShape(inName);    // (1, 3, H, W)
  const auto outDims = engine->getTensorShape(outName);  // (1, C, H, W)
  const int H = inDims.d[2], W = inDims.d[3], C = outDims.d[1];

  // Preprocess: BGR -> RGB, resize, normalize, HWC -> CHW.
  cv::Mat img = cv::imread(argv[2]);
  if (img.empty()) { std::cerr << "cannot read " << argv[2] << "\n"; return 1; }
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, {W, H}, 0, 0, cv::INTER_LINEAR);
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);

  std::vector<float> input(static_cast<size_t>(3) * H * W);
  std::vector<cv::Mat> ch(3);
  cv::split(img, ch);
  for (int c = 0; c < 3; ++c) {
    ch[c] = (ch[c] - kMean[c]) / kStd[c];
    std::memcpy(input.data() + static_cast<size_t>(c) * H * W, ch[c].ptr<float>(),
                sizeof(float) * H * W);
  }

  const size_t inBytes = input.size() * sizeof(float);
  const size_t outCount = static_cast<size_t>(C) * H * W;
  DeviceBuffer dIn(inBytes), dOut(outCount * sizeof(float));
  std::vector<float> logits(outCount);

  cudaMemcpy(dIn.ptr, input.data(), inBytes, cudaMemcpyHostToDevice);
  context->setTensorAddress(inName, dIn.ptr);
  context->setTensorAddress(outName, dOut.ptr);
  if (!context->enqueueV3(nullptr)) { std::cerr << "inference failed\n"; return 1; }
  cudaDeviceSynchronize();
  cudaMemcpy(logits.data(), dOut.ptr, outCount * sizeof(float), cudaMemcpyDeviceToHost);

  // Argmax over channels -> colorized mask.
  cv::Mat mask(H, W, CV_8UC3);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int best = 0;
      float bestV = logits[static_cast<size_t>(y) * W + x];
      for (int c = 1; c < C; ++c) {
        const float v = logits[(static_cast<size_t>(c) * H + y) * W + x];
        if (v > bestV) { bestV = v; best = c; }
      }
      auto& px = mask.at<cv::Vec3b>(y, x);  // BGR out
      px[0] = kPalette[best][2]; px[1] = kPalette[best][1]; px[2] = kPalette[best][0];
    }
  }
  cv::imwrite(argv[3], mask);
  std::cout << "wrote " << argv[3] << "\n";
  return 0;
}
