import tensorflow as tf
print(tf.test.gpu_device_name())
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
