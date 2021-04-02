"""

@author: Martin Käppel
"""

def calculate_weight_vector(y_true, cost_matrix):
    if cost_matrix is None:
        raise Exception('Please provide a cost matrix')
    else:
        number_of_classes = len(cost_matrix[0])
        print('Number of classes: ', number_of_classes)
        print(cost_matrix)
        weight_vector = list()
        for i in range(number_of_classes):
            sum = 0
            for j in range(number_of_classes):
                sum = sum + cost_matrix[i][j]
            weight_vector.append(sum)
        return weight_vector



