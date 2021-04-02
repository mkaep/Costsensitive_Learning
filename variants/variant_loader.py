"""

@author: Martin Käppel
"""

from variants import log_class_weight, balanced_cost, cost_sum, optimized_cost
from support import support
import pickle
import os


def calculate_weight_vector(variant, parameters, y_true, cost_matrix=None):
    if variant == 'BALANCED_COST':
        weight_vector = balanced_cost.calculate_weights(y_true)
        parameters_updated = update_parameters(variant, parameters, weight_vector)
        return weight_vector, parameters_updated
    elif variant == 'BALANCED_COST_LOG':
        weight_vector = log_class_weight.calculate_weights(y_true)
        parameters_updated = update_parameters(variant, parameters, weight_vector)
        return weight_vector, parameters_updated
    elif variant == 'COST_SUM':
        weight_vector = cost_sum.calculate_weight_vector(y_true, cost_matrix)
        parameters_updated = update_parameters(variant, parameters, weight_vector)
        return weight_vector, parameters_updated
    elif variant == 'OPTIMIZED_COST':
        weight_vector = optimized_cost.calculate_weight_vector(y_true, cost_matrix)
        parameters_updated = update_parameters(variant, parameters, weight_vector)
        return weight_vector, parameters_updated
    elif variant == 'APPROXIMATE_COST':
        pass
    elif variant == 'NO':
        parameters_updated = update_parameters(variant, parameters, None)
        return None, parameters_updated
    else:
        raise Exception('This variant does not exists')


def update_parameters(variant, parameters, weight_vector):
    parameters_updated = parameters.copy()
    parameters_updated['model'] = parameters['event_log'] + "_" + variant
    parameters_updated['class_weight_vector'] = weight_vector

    try:
        support.create_json(parameters_updated, os.path.join(parameters['output_folder'], 'model_parameters_CLASS.json'))
    except:
        print(parameters_updated)
        with open(os.path.join(parameters['output_folder'], 'model_parameters_class') + '.pkl', 'wb') as f:
            pickle.dump(parameters_updated, f, pickle.HIGHEST_PROTOCOL)

    return parameters_updated

