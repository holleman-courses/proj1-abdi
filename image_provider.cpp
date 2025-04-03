#include "image_provider.h"
#include "model_settings.h"
#include <TinyMLShield.h>
#include "Arduino.h"

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  static bool camera_initialized = false;
  if (!camera_initialized) {
    if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Camera init failed");
      return kTfLiteError;
    }
    camera_initialized = true;
  }

  byte full_frame[176 * 144];
  Camera.readFrame(full_frame);

  int start_x = (176 - kNumCols) / 2;
  int start_y = (144 - kNumRows) / 2;

  for (int y = 0; y < kNumRows; ++y) {
    for (int x = 0; x < kNumCols; ++x) {
      int src_index = (start_y + y) * 176 + (start_x + x);
      int dst_index = y * kNumCols + x;
      image_data[dst_index] = static_cast<int8_t>(full_frame[src_index] - 128);
    }
  }

  return kTfLiteOk;
}


