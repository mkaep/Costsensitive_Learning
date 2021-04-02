"""

@author: Martin Käppel
"""
import os
import pandas as pd
from lstm import lstm_modified
import tensorflow as tf
from tensorflow.python.client import device_lib


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))


def get_activities_roles(url):
    activities_df = pd.read_csv(os.path.join(url, 'inp_activities.csv'))
    roles_df = pd.read_csv(os.path.join(url, 'inp_roles.csv'))

    ac_index = dict(activities_df.values.tolist())
    rl_index = dict(roles_df.values.tolist())

    return ac_index, rl_index


task = 'evaluation'
parameters = dict()
parameters['base_url'] = 'D:\Costsensitive_Lab\LSTM'
parameters['event_log'] = 'Helpdesk'
parameters['reference_log'] = 'inp_referenceLog'
parameters['output_folder'] = 'output_files\Helpdesk'
parameters['model_type'] = 'shared_cat'
parameters['approach'] = 'DIRECT_APPROACH'
parameters['cost'] = 'BALANCED_COST'
parameters['normalize_weights'] = False

url = os.path.join(parameters['base_url'], parameters['event_log'])

if task == 'embedding':
    # Read activities, roles and pairs
    ac_index, rl_index = get_activities_roles(url)
    pairs_df = pd.read_csv(os.path.join(url, 'inp_pairs.csv'))

    # Train embeddings
    lstm_modified.train_embedding(ac_index, rl_index, pairs_df, parameters)

if task == 'training':
    parameters['l_size'] = 100  # LSTM layer sizes
    parameters['imp'] = 2  # keras lstm implementation 1 cpu,2 gpu
    parameters['lstm_act'] = 'relu'  # optimization function Keras
    parameters['dense_act'] = None  # optimization function Keras
    parameters['optim'] = 'Adam'  # optimization function Keras


    ac_index, rl_index = get_activities_roles(url)
    parameters['ac_index'] = ac_index
    parameters['rl_index'] = rl_index

    batch = list()
    training_task = dict()
    training_task['training_file'] = os.path.join(url, 'inp_log_train_red_10.csv')
    training_task['log_name'] = parameters['event_log']
    batch.append(training_task)

    lstm_modified.train_model(ac_index, rl_index, batch, parameters)

if task == 'evaluation':
    models = list()
    models.append('20210328_0803157160660/model_shared_cat_16-0.69.h5')

    evaluation_tasks = list()
    for m in models:
        evaluation_task = dict()
        evaluation_task['model_file'] = m
        evaluation_task['n_size'] = 5
        evaluation_task['variants'] = [{'imp': 'Random Choice', 'rep': 1}, {'imp': 'Arg Max', 'rep': 1}]
        evaluation_task['test_file'] = os.path.join(url, 'inp_log_test.csv')
        evaluation_tasks.append(evaluation_task)

    lstm_modified.evaluate_model(evaluation_tasks, parameters)