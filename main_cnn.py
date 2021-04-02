"""

@author: Martin Käppel
"""

import cnn_refactored
import tensorflow as tf
import os
from tensorflow.python.client import device_lib


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parameters = dict()
parameters['event_log'] = 'Helpdesk'
parameters['reference_log'] = 'D:\Costsensitive_Lab\CNN\Helpdesk\inp_referenceLog.csv'
parameters['training_file'] = 'D:\Costsensitive_Lab\CNN\Helpdesk\inp_log_train_red_10.csv'
parameters['test_file'] = 'D:\Costsensitive_Lab\CNN\Helpdesk\inp_log_test.csv'
parameters['output_folder'] = 'D:\\'
parameters['approach'] = 'OUTPUT_SHIFT'
parameters['cost'] = 'BALANCED_COST'
parameters['normalize_weights'] = False

print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

cnn_refactored.main(parameters, 1)
