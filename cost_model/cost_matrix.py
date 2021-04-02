import numpy as np
"""

@author: Martin Käppel
"""

def calculate_cost_matrix(reference_y_enc):
    reference_y_enc_list = list(reference_y_enc)
    reference_label_counted = dict()
    labels = set(reference_y_enc_list)
    for act in labels:
        reference_label_counted.update({act: reference_y_enc_list.count(act)})

    number_of_classes = len(labels)
    cost_matrix = np.zeros((number_of_classes, number_of_classes))
    for i in reference_label_counted:
        for j in reference_label_counted:
            if i != j:
                if reference_label_counted[i] <= reference_label_counted[j]:
                    cost_matrix[i, j] = reference_label_counted[j]/reference_label_counted[i]
                    cost_matrix[j, i] = 1
    return cost_matrix
