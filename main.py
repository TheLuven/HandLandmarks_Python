import Core.Core as ie
import Core.InferenceProcess as ip
import tensorflow as tf
#

if __name__ == "__main__":
    ie.version()
    ip.initProcess(5000000,"models/palm_detection_full.tflite","models/anchors.csv")

