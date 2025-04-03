// detection_responder.cpp
#include "detection_responder.h"
#include "Arduino.h"

void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        int8_t hand_score, int8_t no_hand_score) {
  static bool initialized = false;
  if (!initialized) {
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    initialized = true;
  }

  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);
  delay(100);
  digitalWrite(LEDB, HIGH);

  if (hand_score > no_hand_score) {
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Hand score: %d Non-hand score: %d",
                       hand_score, no_hand_score);
}
