"""

@author: Martin Käppel
"""
import numpy as np
from sklearn.metrics import classification_report, \
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score, \
    matthews_corrcoef, hinge_loss, roc_auc_score, \
    accuracy_score, f1_score, fbeta_score, hamming_loss, jaccard_score, log_loss, precision_score, recall_score, \
    zero_one_loss
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score, make_index_balanced_accuracy


def evaluate(data, weight_vector=None):
    ac_expect = list(data['ac_expect'])
    ac_pred = list(data['ac_pred'])
    ac_pred_max = list()

    if weight_vector is None:
        for i in ac_pred:
            ac_pred_max.append(np.argmax(i))
    else:
        # Variant Output Shift
        predicted_shift = list()
        for e in ac_pred:
            predicted_shift.append(shift_output(e, weight_vector))
        for i in predicted_shift:
            ac_pred_max.append(np.argmax(i))

    print("F-Score (Macro): ", f1_score(ac_expect, ac_pred_max, average='macro'))
    print("Precision (Macro): ", precision_score(ac_expect, ac_pred_max, average='macro'))
    print("Recall (Macro): ", recall_score(ac_expect, ac_pred_max, average='macro'))
    print("Geometric Mean (Macro): ", geometric_mean_score(ac_expect, ac_pred_max, average='macro'))
    print(classification_report(ac_expect, ac_pred_max))


def normalize_vector(vector):
    sum = 0
    for i in vector:
        sum = sum + i

    normalized_vector = list()
    for i in vector:
        normalized_vector.append(i/sum)
    return normalized_vector


def shift_output(output_vector, weight_vector):
    number_of_classes = len(output_vector)
    shifted_output = list()
    for i in range(number_of_classes):
        sum = 0
        for j in range(number_of_classes):
            sum = sum + output_vector[i]*weight_vector[i]
        shifted_output.append(sum)
    shifted_output = normalize_vector(shifted_output)
    return shifted_output




