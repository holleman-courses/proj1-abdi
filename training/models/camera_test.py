import serial
import serial.tools.list_ports
import numpy as np
import cv2
import sys
import time
import tensorflow as tf
from PIL import Image

# Model settings
MODEL_PATH = "hand_detector_model_quant.tflite"
IMG_WIDTH = 160
IMG_HEIGHT = 120
CROP_SIZE = 96
BAUD = 115200

def detect_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Arduino" in port.description or "nano33ble" in port.description or "usbmodem" in port.device:
            return port.device
    return None

def preprocess_image(frame):
    # Crop center 96x96 from 160x120
    x_start = (IMG_WIDTH - CROP_SIZE) // 2
    y_start = (IMG_HEIGHT - CROP_SIZE) // 2
    cropped = frame[y_start:y_start+CROP_SIZE, x_start:x_start+CROP_SIZE]

    # Normalize + quantize to int8
    normalized = cropped.astype(np.float32) / 255.0
    standardized = (normalized - 0.5) * 2.0
    quantized = (standardized * 127).astype(np.int8)
    return quantized.reshape(1, 96, 96, 1)

def predict(img_array, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print("🧠 Raw Output:", output)

    if output[0][0] > 0:
        print(" Hand Detected!")
    else:
        print(" No Hand Detected.")

def main():
    # Load model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Loaded model ")
    print("Input shape:", input_details[0]['shape'])

    port = detect_port()
    if not port:
        print("Arduino not found.")
        return

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, BAUD, timeout=2)
    time.sleep(2)

    print("Waiting for Arduino ready signal...")
    while True:
        line = ser.readline().decode().strip()
        if line == "READY":
            print("Arduino ready ")
            break

    try:
        while True:
            cmd = input("Press Enter to capture (or 'q' to quit): ")
            if cmd.lower() == 'q':
                break

            ser.write(b'c')  # Send capture command
            ack = ser.readline().decode().strip()
            if ack != "CAPTURING":
                print("c Capture not acknowledged")
                continue

            print(" Receiving frame...")
            data = bytearray()
            start_time = time.time()
            IMG_SIZE = IMG_WIDTH * IMG_HEIGHT

            while len(data) < IMG_SIZE:
                chunk = ser.read(IMG_SIZE - len(data))
                if not chunk and (time.time() - start_time > 3):
                    print("Timeout during capture.")
                    break
                data.extend(chunk)
                print(f"\r{len(data)}/{IMG_SIZE} bytes", end='')

            print()
            if len(data) == IMG_SIZE:
                frame = np.frombuffer(data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH))
                cv2.imshow("Captured Frame", frame)
                cv2.waitKey(1)

                img_array = preprocess_image(frame)
                predict(img_array, interpreter, input_details, output_details)

            else:
                print(" Incomplete frame")

    finally:
        ser.close()
        cv2.destroyAllWindows()
        print("Serial closed.")

if __name__ == "__main__":
    main()
