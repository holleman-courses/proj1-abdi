#ifndef HAND_DETECTION_MODEL_SETTINGS_H_
#define HAND_DETECTION_MODEL_SETTINGS_H_

constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;
constexpr int kImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;
constexpr int kHandIndex = 1;
constexpr int kNoHandIndex = 0;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // HAND_DETECTION_MODEL_SETTINGS_H_
