#include <iostream>
#include <vector>

#include "paddle_inference_api.h"

#include <glog/logging.h>

namespace paddle {

void RunNative(int batch_size, const std::string &model_dirname) {
  // 创建参数
  NativeConfig config{};
  config.use_gpu = false;
  config.SetCpuMathLibraryNumThreads(1);

  // 设置模型的参数路径
  config.model_dir = model_dirname;

  // 当模型输入是多个的时候，这个配置是必要的。
  config.specify_input_name = true;

  auto predictor = CreatePaddlePredictor(config);

  float *data = new float[4];
  *(data + 0) = 9;
  *(data + 1) = 5;
  *(data + 2) = 2;
  *(data + 3) = 10;

  PaddleTensor tensor;
  tensor.name = "x";
  tensor.shape = std::vector<int>({1, 4});
  tensor.data = PaddleBuf(static_cast<void *>(data), sizeof(float) * 4);
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  // 输出
  std::vector<PaddleTensor> outputs;
  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);

  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  auto *data_out = static_cast<float *>(outputs.front().data.data());

  LOG(INFO) << "data_out: " << *(data_out);
}
}  // namespace paddle

// bin/start_simple_lr ../../python-scripts/train_parameters

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "need provide train dir.";
    return EXIT_FAILURE;
  }

  std::string model_dir = std::string(argv[1]);
  LOG(INFO) << "using train_parameter dir: " << model_dir;

  paddle::RunNative(1, model_dir);
  return EXIT_SUCCESS;
}
