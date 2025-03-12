import tensorflow as tf
import os

def main():
    # Update this path to where your data folder is located.
    # If your folder structure is as expected, you can use:
    data_dir = "data"  
    # OR use the absolute path:
    # data_dir = "/Users/abdirahimahmed/Desktop/hw4_iot/data"
    
    print("Using data directory:", os.path.abspath(data_dir))
    
    # Create the training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(64, 64),
        batch_size=2
    )
    
    # Create the validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(64, 64),
        batch_size=2
    )
    
    # Build a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )
    
    # Save the model as model.h5 in the current directory
    model.save("model.h5")
    print("Model has been trained and saved as 'model.h5'.")

if __name__ == "__main__":
    main()
