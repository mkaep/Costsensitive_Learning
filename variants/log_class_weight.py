"""

@author: Martin Käppel
"""
import numpy as np
import math


def calculate_weights(y_true):
    reference_y = np.argmax(y_true, axis=1)

    # Calculate with sklearn
    total = np.sum(list(reference_y))
    class_weights = dict()

    labels = set(np.unique(reference_y))
    labels_counted = dict()

    y_list = list(reference_y)
    for i in labels:
        labels_counted.update({i: y_list.count(i)})

    for k in labels:
        score = math.log(0.15*total/float(labels_counted[k]))
        class_weights[k] = score if score > 1.0 else 1.0


    return class_weights


