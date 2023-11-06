import tensorflow as tf


if __name__ == "__main__":
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    print(f"TensorFlow version: {tf.__version__}")

