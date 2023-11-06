import tensorflow as tf
import numpy as np
import cv2

def version():
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
    print(f"TensorFlow version: {tf.__version__}")
class Core:
    def __init__(self, model):
        self.model = model
        self.interpreter = tf.lite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(self.input_details)
        print(self.output_details)




    def inputImage(self,image,mean=0, divide=255,color = "RGB"):
        if color == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_shape = self.input_details[0]['shape']
        image = cv2.resize(image, (input_shape[2], input_shape[1]))
        # Convert the image to a NumPy array and normalize it if needed
        image = np.asarray(image, dtype=np.float32)
        image -= mean  # Normalize pixel values to [0, 1] (if needed)
        image /= divide  # Normalize pixel values to [0, 1] (if needed)
        # Add a batch dimension to the image
        image = np.expand_dims(image, axis=0)

        # Set the input tensor to the preprocessed image
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

    def infer(self):
        self.interpreter.invoke()

    def getData(self,index=0):
        return self.interpreter.get_tensor(self.output_details[index]['index'])