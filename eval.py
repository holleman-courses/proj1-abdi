import os
import tensorflow as tf

def main():
    # Print the current working directory
    cwd = os.getcwd()
    print("Current working directory:", cwd)
    
    # Define the expected path to model.h5
    model_path = "model.h5"
    
    # Check if model.h5 exists
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found in {cwd}.")
        print("Make sure you have run train.py and that model.h5 is saved in this directory.")
        return

    # 1. Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load the dataset again for evaluation
    data_dir = "data"  # same directory as used in train.py
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(64, 64),
        batch_size=1,  # can be 1 or 2, etc.
        shuffle=False  # we want consistent ordering for analysis
    )
    
    # 3. Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test Loss: {loss:.3f}")

if __name__ == "__main__":
    main()

