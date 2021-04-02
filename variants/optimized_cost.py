"""

@author: Martin Käppel
"""

import numpy as np


def calculate_weight_vector(y_true, cost_matrix):
    if cost_matrix is None:
        raise Exception('Please provide a cost matrix')
    else:
        nc = len(cost_matrix[0])
        # Create coefficient matrix
        coefficient_matrix = calculate_coefficient_matrix(cost_matrix, nc)

        weight_vector = list()
        if calculate_rank(coefficient_matrix) < nc:
            print("Consistent cost matrix. Start deriving an optimal weight vector...")
            # Solve equation system
            return weight_vector
        else:
            print("Cost matrix is not consistent. Could not derive an optimal weight vector. Try approximate cost or "
                  "any other option. Train without cost-sensitive methods.")
            return None


def calculate_coefficient_matrix(cost_matrix, number_of_classes):
    coefficient_matrix = np.zeros((int((number_of_classes * (number_of_classes - 1)) / 2), number_of_classes))
    indent = 0
    for i in range(1, number_of_classes):
        coefficient_matrix, indent = determine_coefficient(coefficient_matrix, cost_matrix, i + 1, i, number_of_classes, indent)
    return coefficient_matrix


def determine_coefficient(coefficient_matrix, cost_matrix, start, column, nc, indent):
    j = 1
    for i in range(start, nc + 1):
        coefficient_matrix[indent + i - start, column - 1] = cost_matrix[i - 1, column - 1]
        coefficient_matrix[indent + i - start, column - 1 + j] = (-1) * cost_matrix[column - 1, i - 1]
        j = j + 1
    indent = indent + nc+1-start
    return coefficient_matrix, indent


def calculate_rank(matrix, eps=1e-200000000):
    u, s, vh = np.linalg.svd(matrix)
    return len([x for x in s if abs(x) > eps])

