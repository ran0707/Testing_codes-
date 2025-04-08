import tensorflow as tf

print(tf.__version__)

print('Ava GPU:',tf.config.list_physical_devices('GPU'))