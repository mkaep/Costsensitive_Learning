"""

@author: Martin Käppel
"""
import pandas as pd
import csv
import numpy as np
import os
import math
import itertools
import json

from approach.lstm import embedding_training as et
from cost_model import cost_matrix as cm
from approach.lstm import support as sup
from approach.lstm import examples_creator as exc
from approach.lstm import model_loader as mload
from approach.lstm import model_predictor as pr
from variants import variant_loader
from analyzer import imbalance


def load_embedded(index, filename, directory):
    """Loading of the embedded matrices.
    parms:
        index (dict): index of activities or roles.
        filename (str): filename of the matrix file.
    Returns:
        numpy array: array of weights.
    """
    weights = list()
    input_folder = directory
    with open(os.path.join(input_folder, filename), 'r') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in filereader:
            cat_ix = int(row[0])
            if index[cat_ix] == row[1].strip():
                weights.append([float(x) for x in row[2:]])
        csvfile.close()
    return np.array(weights)


def train_embedding(ac_index, rl_index, pairs_df, parameters):
    index_ac = {v: k for k, v in ac_index.items()}
    index_rl = {v: k for k, v in rl_index.items()}

    pairs = list()
    for i in range(0, len(pairs_df)):
        pairs.append((pairs_df.iloc[i]['Activity'], pairs_df.iloc[i]['Role']))

    # Calculate dimensions for the embedding vectors
    dim_number = math.ceil(len(list(itertools.product(*[list(ac_index.items()), list(rl_index.items())]))) ** 0.25)

    # Create Model for training the embedding
    model = et.ac_rl_embedding_model(ac_index, rl_index, dim_number)
    model.summary()

    n_positive = 1024
    gen = et.generate_batch(pairs, ac_index, rl_index, n_positive, negative_ratio=2)

    # Train
    model.fit_generator(gen, epochs=100, steps_per_epoch=len(pairs) // n_positive, verbose=2)

    # Extract embeddings
    ac_layer = model.get_layer('activity_embedding')
    rl_layer = model.get_layer('role_embedding')

    ac_weights = ac_layer.get_weights()[0]
    rl_weights = rl_layer.get_weights()[0]

    print(os.path.join(parameters['folder'], 'ac_' + parameters['event_log'] + '.emb'))
    sup.create_file_from_list(et.reformat_matrix(index_ac, ac_weights),
                              os.path.join(parameters['folder'],
                                           'ac_' + parameters['event_log'] + '.emb'))
    sup.create_file_from_list(et.reformat_matrix(index_rl, rl_weights),
                              os.path.join(parameters['folder'],
                                           'rl_' + parameters['event_log'] + '.emb'))


def train_model(ac_index, rl_index, training_logs, parameters):
    index_ac = {v: k for k, v in ac_index.items()}
    index_rl = {v: k for k, v in rl_index.items()}

    for training_log in training_logs:
        i = 0
        training_log_df = pd.read_csv(training_log['training_file'], sep=",")
        # Train the model
        # Create examples
        seq_creator = exc.SequencesCreator(training_log_df, ac_index, rl_index)
        examples = seq_creator.vectorize('shared_cat')
        labels = examples['next_evt']['activities']

        labels_enc = np.argmax(labels, axis=1)
        imbalance.calculate_imbalance_degree(labels_enc)

        # Depending on the variant calculate a cost matrix
        if parameters['cost'] == 'COST_SUM' or parameters['cost'] == 'OPTIMIZED_COST' or parameters[
            'cost'] == 'APPROXIMATE_COST':
            cost_matrix = cm.calculate_cost_matrix(labels_enc)
            print(cost_matrix)
        else:
            cost_matrix = None

        class_weights, parameters = variant_loader.calculate_weight_vector(parameters['cost'], parameters,
                                                                           labels, cost_matrix)

        # Load embedded matrix
        ac_weights = load_embedded(index_ac, 'ac_' + training_log['log_name'] + '.emb', parameters['output_folder'])
        rl_weights = load_embedded(index_rl, 'rl_' + training_log['log_name'] + '.emb', parameters['output_folder'])

        folder_id = sup.folder_id() + str(i)
        output_folder = os.path.join('output_files', folder_id)

        # Method Export Params in model_trainer modificated
        # Export params
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            os.makedirs(os.path.join(output_folder, 'parameters'))

        m_loader = mload.ModelLoader(parameters)
        m_loader.train('shared_cat', examples, ac_weights, rl_weights, output_folder, class_weights)

        parameters['dim'] = dict(
                samples=examples['prefixes']['activities'].shape[0],
                time_dim=examples['prefixes']['activities'].shape[1],
                features=len(ac_index))
        parameters['training_task'] = training_log
        export_parameters(parameters, os.path.join(output_folder, 'parameters'))
        i = i+1


def export_parameters(parameters, path):
    sup.create_json(parameters, os.path.join(path, 'model_parameters.json'))
    print(parameters)


def load_parameters(path_parameters):
    # Loading of parameters from training
    path = os.path.join(path_parameters,
                            'parameters',
                            'model_parameters.json')
    with open(path) as file:
        data = json.load(file)
    return data


def evaluate_model(evaluation_tasks, parameters):
    for ev in evaluation_tasks:
        test_log_df = pd.read_csv(ev['test_file'], sep=",")
        parameters = load_parameters(os.path.join(parameters['output_folder'], ev['model_file'].split('/')[0]))
        parameters['model_file'] = ev['model_file']
        parameters['variants'] = ev['variants']
        pr.ModelPredictor(parameters, test_log_df, parameters['ac_index'], parameters['rl_index'])
