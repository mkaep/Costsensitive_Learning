"""

@author: Martin Käppel
"""

import pandas as pd
import numpy as np
import os
import collections

from cost_model import cost_matrix as cm
from approach.cnn import example_creator
from approach.cnn.models import model_3_layer
from analyzer import analyzer, imbalance
from util import plotting
from evaluator import evaluator_dict as ev
from sklearn import preprocessing
from keras.utils import np_utils
from variants import variant_loader
from support import support


def normalize_vector(vector):
    sum = 0
    for i in vector:
        sum = sum + i

    normalized_vector = list()
    for i in vector:
        normalized_vector.append(i / sum)
    return normalized_vector


def process_reference_log(parameters, verbose):
    output_path = os.path.join(parameters['output_folder'], parameters['event_log'])

    if not os.path.exists(output_path):
        os.mkdir(os.path.join(parameters['output_folder'], parameters['event_log']))
    else:
        print("The directory for the event log already exists!")

    parameters['output_folder'] = output_path

    reference_log_df = analyzer.load_reference_log(parameters['reference_log'])
    max_trace_length, n_caseid, n_activity, activities = analyzer.prescriptive_analysis(reference_log_df)

    parameters['max_trace_length'] = max_trace_length
    parameters['n_caseid'] = n_caseid
    parameters['n_activities'] = n_activity
    parameters['activities'] = activities

    if verbose > 1:
        # Print the distribution of the activities and store the plot in the output folder
        activities_counted = analyzer.get_activity_distribution(reference_log_df, activities)
        plotting.plot_barchart_from_dictionary(activities_counted,
                                               "Activity Distribution Reference Log (" + parameters['event_log'] + ")",
                                               "Activity", "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'])

    # Extract labels (i.e. names of activities that occur as labels) for the data set and encode them
    reference_y = example_creator.get_label(reference_log_df.groupby('CaseID').agg({'Activity': lambda x: list(x)}))

    # Calculate imbalance degree
    imbalance.calculate_imbalance_degree(reference_y)

    if verbose > 1:
        # Print the distribution of the labels of the reference log
        labels_counted = analyzer.get_label_distribution(reference_y, set(reference_y))
        plotting.plot_barchart_from_dictionary(labels_counted,
                                               "Label Distribution Reference Log (" + parameters['event_log'] + ")",
                                               "Label", "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'])

    # Encode the labels extracted from the reference log and export them to the output folder
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(reference_y)
    support.export_encoding(parameters['output_folder'], label_encoder)
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    parameters['encoding'] = le_name_mapping
    if verbose > 1:
        print("Encoding Mapping: ")
        print(parameters['encoding'])

    # Encode the the reference training samples
    reference_y_enc = label_encoder.transform(reference_y)

    # Depending on the variant calculate a cost matrix
    if parameters['cost'] == 'COST_SUM' or parameters['cost'] == 'OPTIMIZED_COST' or parameters[
        'cost'] == 'APPROXIMATE_COST':
        cost_matrix = cm.calculate_cost_matrix(reference_y_enc)
        print(cost_matrix)
    else:
        cost_matrix = None

    reference_y_enc = np.asarray(reference_y_enc)
    reference_y_one_hot = np_utils.to_categorical(reference_y_enc, label_encoder.classes_.size)

    all_labels_enc = set(reference_y_enc)
    parameters['labels_enc'] = all_labels_enc
    parameters['labels'] = set(reference_y)

    return reference_y_one_hot, cost_matrix, parameters


def main(parameters, verbose):
    # Preprocessing reference log to get all necessary information that get maybe lost in training and test data
    reference_y_one_hot, cost_matrix, parameters = process_reference_log(parameters, verbose)

    # Read training and test log
    training_log_df = pd.read_csv(parameters['training_file'], sep=",")
    test_log_df = pd.read_csv(parameters['test_file'], sep=",")

    if verbose > 1:
        # Plot the distribution of the activities in training and test data
        activities_training_counted = analyzer.get_activity_distribution(training_log_df, parameters['activities'])
        activities_test_counted = analyzer.get_activity_distribution(test_log_df, parameters['activities'])

        plotting.plot_barchart_from_dictionary(activities_training_counted,
                                               "Activity Distribution Training Log (" + parameters['event_log'] + ")",
                                               "Activity", "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'])
        plotting.plot_barchart_from_dictionary(activities_test_counted,
                                               "Activity Distribution Test Log (" + parameters['event_log'] + ")",
                                               "Activity", "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'])

    # Group activities and timestamp within a trace
    counted = collections.Counter(test_log_df['CaseID'])
    print(counted)
    list_to_remove = list()
    for i in counted:
        if counted[i] < 2:
            list_to_remove.append(i)

    print(list_to_remove)

    training_act = training_log_df.groupby('CaseID').agg({'Activity': lambda x: list(x)})

    training_time = training_log_df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

    test_act = test_log_df.groupby('CaseID').agg({'Activity': lambda x: list(x)})
    test_time = test_log_df.groupby('CaseID').agg({'Timestamp': lambda x: list(x)})

    # Create labeled data set for training
    train_x = example_creator.get_image(training_act, training_time, parameters['max_trace_length'],
                                        parameters['n_activities'], parameters['activities'])
    test_x = example_creator.get_image(test_act, test_time, parameters['max_trace_length'],
                                       parameters['n_activities'], parameters['activities'])

    train_y = example_creator.get_label(training_act)
    test_y = example_creator.get_label(test_act)

    if verbose > 1:
        # Print the distribution of the labels of the training and test log
        label_training_counted = analyzer.get_label_distribution(train_y, parameters['labels'])
        label_test_counted = analyzer.get_label_distribution(test_y, parameters['labels'])

        plotting.plot_barchart_from_dictionary(label_training_counted,
                                               "Label Distribution Training Log (" + parameters['event_log'] + ")",
                                               "Label", "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'], encode=False, encoder=None)
        plotting.plot_barchart_from_dictionary(label_test_counted,
                                               "Label Distribution Test Log (" + parameters['event_log'] + ")", "Label",
                                               "Number of Occurrence", save=True,
                                               output_file=parameters['output_folder'], encode=False, encoder=None)

    # Load encoder and encode the labels of the training and test log
    label_encoder = support.load_encoding(parameters['output_folder'])
    train_y = label_encoder.transform(train_y)
    test_y = label_encoder.transform(test_y)

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    test_x = np.asarray(test_x)
    test_y = np.asarray(test_y)

    train_y_one_hot = np_utils.to_categorical(train_y, label_encoder.classes_.size)
    test_y_one_hot = np_utils.to_categorical(test_y, label_encoder.classes_.size)

    class_weights, parameters = variant_loader.calculate_weight_vector(parameters['cost'], parameters,
                                                                       reference_y_one_hot, cost_matrix)
    print("Class weights:")
    print(class_weights)
    if parameters['approach'] == 'OUTPUT_SHIFT' or parameters['approach'] == 'NO':
        model = model_3_layer.training_model(parameters['max_trace_length'], parameters['n_activities'],
                                             label_encoder.classes_.size, train_x, train_y_one_hot, parameters['model'],
                                             None, parameters['output_folder'])
    elif parameters['approach'] == 'DIRECT_APPROACH':
        if parameters['normalize_weights'] == False:
            model = model_3_layer.training_model(parameters['max_trace_length'], parameters['n_activities'],
                                                 label_encoder.classes_.size, train_x, train_y_one_hot,
                                                 parameters['model'],
                                                 class_weights, parameters['output_folder'])
        else:
            class_weights = normalize_vector(class_weights)
            class_weights_dic = dict()
            for i in range(len(class_weights)):
                class_weights_dic[i] = class_weights[i]

            model = model_3_layer.training_model(parameters['max_trace_length'], parameters['n_activities'],
                                                 label_encoder.classes_.size, train_x, train_y_one_hot,
                                                 parameters['model'],
                                                 class_weights_dic, parameters['output_folder'])
    else:
        raise Exception('This approach does not exists')

    # Evaluation
    evaluated_models = dict()
    if parameters['approach'] == 'OUTPUT_SHIFT':
        evaluated_models.update(
            {str(parameters['model']): ev.evaluate(model, test_x, test_y_one_hot, parameters['output_folder'],
                                                   parameters['model'],
                                                   all_labels=parameters['labels_enc'],
                                                   weight_vector=class_weights)})
    elif parameters['approach'] == 'DIRECT_APPROACH' or parameters['approach'] == 'NO':
        evaluated_models.update(
            {str(parameters['model']): ev.evaluate(model, test_x, test_y_one_hot, parameters['output_folder'],
                                                   parameters['model'],
                                                   all_labels=parameters['labels_enc'])})
    else:
        raise Exception('This approach does not exists')

    df = pd.DataFrame(evaluated_models)
    df.to_csv(os.path.join(parameters['output_folder'], 'evaluation_summary.csv'), sep=' ', header=True, mode='a')
