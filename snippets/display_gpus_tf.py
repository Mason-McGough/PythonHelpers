"""
Short code to list all GPU and CPU devices recognized by Tensorflow.
"""

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
