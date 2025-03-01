#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "keystroke_model.h"
#include "keystroke.h"

namespace {
  // モデル変数
  const tflite::Model* model = ::tflite::GetModel(keystroke_model_tflite);

  // Tensor Arena (16バイトアライン)
  constexpr int kTensorArenaSize = 32 * 1024;
  __attribute__((aligned(16))) uint8_t tensor_arena[kTensorArenaSize];

  // 演算子リゾルバー - 必要最小限の演算子を登録
  static tflite::MicroMutableOpResolver<4> resolver;

  // インタープリターのポインタ
  tflite::MicroInterpreter* interpreter = nullptr;
}

// キーストロークdynamicsモデルの初期化
bool keystroke_init(void) {
  // 最小限の演算子のみを登録
  resolver.AddFullyConnected();

  // インタープリターの作成
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );
  interpreter = &static_interpreter;

  // テンソルの割り当て
  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    return false;
  }

  return true;
}

// 推論を行う関数（内部実装） - float版
float keystroke_infer_float(float* input_data, int input_length) {
  if (!interpreter) {
    return -1.0f;
  }

  // 入力テンソルを取得
  TfLiteTensor* input = interpreter->input(0);

  // 入力データをコピー
  if (input->bytes / sizeof(float) < input_length) {
    return -1.0f;
  }

  for (int i = 0; i < input_length; i++) {
    input->data.f[i] = input_data[i];
  }

  // 推論を実行
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    return -1.0f;
  }

  // 出力を取得
  TfLiteTensor* output = interpreter->output(0);
  float prediction = output->data.f[0];

  return prediction;
}

int keystroke_predict(int* features, int feature_count) {
    int sum = 0;
    for (int i = 0; i < feature_count; i++) {
        sum += features[i];
    }

    const int threshold = 500 * feature_count;

    if (sum < threshold) {
        return 500;
    } else {
        return 800;
    }

}

extern "C" {
    int __cxa_guard_acquire(long long* guard) { return 1; }
    void __cxa_guard_release(long long* guard) {}
    void __cxa_guard_abort(long long* guard) {}
    int __cxa_atexit(void (*func)(void*), void* arg, void* dso) { return 0; }
    void __stack_chk_fail() { while(1); }
    void _Unwind_Resume() { while(1); }
}
