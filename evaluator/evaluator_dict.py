"""

@author: Martin Käppel
"""
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, \
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score, \
    matthews_corrcoef, hinge_loss, roc_auc_score, \
    accuracy_score, f1_score, fbeta_score, hamming_loss, jaccard_score, log_loss, precision_score, recall_score, \
    zero_one_loss
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score, make_index_balanced_accuracy


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


def evaluate(model, test_x, test_y, output_folder, title, class_specific=False, all_labels=None, weight_vector=None):
    # Ensure that labels is an np array
    if all_labels is None:
        labels = None
    else:
        labels = list(all_labels)

    if weight_vector is None:
        y_predicted = model.predict(test_x)
        y_predicted_max = np.argmax(y_predicted, axis=1)
    else:
        # Variant Output Shift
        y_predicted = model.predict(test_x)
        predicted_shift = list()
        for e in y_predicted:
            predicted_shift.append(shift_output(e, weight_vector))
        y_predicted_max = np.argmax(predicted_shift, axis=1)

    y_test_max = np.argmax(test_y, axis=1)

    # Print classification report
    report = classification_report(y_test_max, y_predicted_max, labels=labels, output_dict=True, digits=5)
    report_df = pd.DataFrame(report)
    report_df.to_csv(os.path.join(output_folder, 'report_'+title+'.csv'), sep=' ', header=True, mode='a')


    # Print confusion matrix
    cm = confusion_matrix(y_test_max, y_predicted_max, labels=labels)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(output_folder, 'cm_'+title+'.csv'), sep=' ', header=True, mode='a')

    metrics = dict()

    # Evaluate further metrics
    # =============================================================================
    #    Balanced Accuracy Score
    # =============================================================================
    metrics['Balanced Accuracy Score'] = balanced_accuracy_score(y_test_max, y_predicted_max)

    # =============================================================================
    #    Cohen Kappa Score
    # =============================================================================
    metrics['Cohen Kappa Score (No weighted)'] = cohen_kappa_score(y_predicted_max, y_test_max, weights=None)
    metrics['Cohen Kappa Score (Linear weighted)'] = cohen_kappa_score(y_predicted_max, y_test_max, weights='linear')
    metrics['Cohen Kappa Score (Quadratic weighted)'] = cohen_kappa_score(y_predicted_max, y_test_max,
                                                                          weights='quadratic')

    # =============================================================================
    #    Hinge Loss
    # =============================================================================
    metrics['Hinge Loss'] = hinge_loss(y_test_max, y_predicted, labels=labels)

    # =============================================================================
    #    Matthews Correlation Coefficient
    # =============================================================================
    metrics['Matthews Correlation Coefficient'] = matthews_corrcoef(y_test_max, y_predicted_max)

    # =============================================================================
    #    Top k Accuracy Score (does not work, To DO)
    # =============================================================================
    # print("\n Top k Accuracy: ")
    # print(top_k_accuracy_score(y_test_max, y_predicted_max, k=5))

    # =============================================================================
    #    The following also work in the multi label case
    # =============================================================================

    # =============================================================================
    #    Accuracy Score
    # =============================================================================
    metrics['Accuracy Score'] = accuracy_score(y_test_max, y_predicted_max)

    # =============================================================================
    #    F1 Score
    # =============================================================================
    metrics['F Score (Micro)'] = f1_score(y_test_max, y_predicted_max, average='micro')
    metrics['F Score (Macro)'] = f1_score(y_test_max, y_predicted_max, average='macro')
    metrics['F Score (Weighted)'] = f1_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['F Score (None, i.e. for each class)'] = f1_score(y_test_max, y_predicted_max, average=None)

    # =============================================================================
    #    ROC AUC Score (in case of multi class sklearn only support macro and weighted averages)
    # =============================================================================
    # ROC AUC only works if each label occurs at least one time. Hence, we need to catch a exception
    print(y_test_max)
    try:
        metrics['ROC AUC Score (OVR) Macro'] = roc_auc_score(y_test_max, y_predicted, multi_class='ovr',
                                                             average='macro', labels=labels)
        metrics['ROC AUC Score (OVR) Weighted'] = roc_auc_score(y_test_max, y_predicted, multi_class='ovr',
                                                                average='weighted', labels=labels)
        metrics['ROC AUC Score (OVO) Macro'] = roc_auc_score(y_test_max, y_predicted, multi_class='ovo',
                                                             average='macro', labels=labels)
        metrics['ROC AUC Score (OVO) Weighted'] = roc_auc_score(y_test_max, y_predicted, multi_class='ovo',
                                                                average='weighted', labels=labels)
    except:
        print("Cannot calculate ROC AUC Score!")
        pass


    # =============================================================================
    #    F Beta Score
    # =============================================================================
    metrics['F Beta Score (Micro) b=0.5'] = fbeta_score(y_test_max, y_predicted_max, average='micro', beta=0.5)
    metrics['F Beta Score (Macro) b=0.5'] = fbeta_score(y_test_max, y_predicted_max, average='macro', beta=0.5)
    metrics['F Beta Score (Weighted) b=0.5'] = fbeta_score(y_test_max, y_predicted_max, average='weighted', beta=0.5)
    if class_specific:
        metrics['F Beta Score (None, i.e. for each class) b=0.5'] = fbeta_score(y_test_max, y_predicted_max, average=None,
                                                                            beta=0.5)

    metrics['F Beta Score (Micro) b=1.5'] = fbeta_score(y_test_max, y_predicted_max, average='micro', beta=1.5)
    metrics['F Beta Score (Macro) b=1.5'] = fbeta_score(y_test_max, y_predicted_max, average='macro', beta=1.5)
    metrics['F Beta Score (Weighted) b=1.5'] = fbeta_score(y_test_max, y_predicted_max, average='weighted', beta=1.5)
    if class_specific:
        metrics['F Beta Score (None, i.e. for each class) b=1.5'] = fbeta_score(y_test_max, y_predicted_max, average=None,
                                                                            beta=1.5)

    # =============================================================================
    #    Hamming Loss
    # =============================================================================
    metrics['Hamming Loss'] = hamming_loss(y_test_max, y_predicted_max)

    # =============================================================================
    #    Jaccard Score
    # =============================================================================
    metrics['Jaccard Score (Micro)'] = jaccard_score(y_test_max, y_predicted_max, average='micro')
    metrics['Jaccard Score (Macro)'] = jaccard_score(y_test_max, y_predicted_max, average='macro')
    metrics['Jaccard Score (Weighted)'] = jaccard_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Jaccard Score (None, i.e. for each class)'] = jaccard_score(y_test_max, y_predicted_max, average=None)

    # =============================================================================
    #    Log Loss
    # =============================================================================
    metrics['Logg Loss'] = log_loss(y_test_max, y_predicted, labels=labels)

    # =============================================================================
    #    Precision Score
    # =============================================================================
    metrics['Precision Score (Micro)'] = precision_score(y_test_max, y_predicted_max, average='micro')
    metrics['Precision Score (Macro)'] = precision_score(y_test_max, y_predicted_max, average='macro')
    metrics['Precision Score (Weighted)'] = precision_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Precision Score (None, i.e. for each class)'] = precision_score(y_test_max, y_predicted_max, average=None)

    # =============================================================================
    #    Specificity Score
    # =============================================================================
    metrics['Specificity Score (Micro)'] = specificity_score(y_test_max, y_predicted_max, average='micro')
    metrics['Specificity Score (Macro)'] = specificity_score(y_test_max, y_predicted_max, average='macro')
    metrics['Specificity Score (Weighted)'] = specificity_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Specificity Score (None, i.e. for each class)'] = specificity_score(y_test_max, y_predicted_max, average=None)

    # =============================================================================
    #    Recall Score (also named Sensitivity Score). Hence, the Sensitivity Score values
    #   should be the same as the Recall Score values
    # =============================================================================
    metrics['Recall Score (Micro)'] = recall_score(y_test_max, y_predicted_max, average='micro')
    metrics['Recall Score (Macro)'] = recall_score(y_test_max, y_predicted_max, average='macro')
    metrics['Recall Score (Weighted)'] = recall_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Recall Score (None, i.e. for each class)'] = recall_score(y_test_max, y_predicted_max, average=None)

    metrics['Sensitivity Score (Micro)'] = sensitivity_score(y_test_max, y_predicted_max, average='micro')
    metrics['Sensitivity Score (Macro)'] = sensitivity_score(y_test_max, y_predicted_max, average='macro')
    metrics['Sensitivity Score (Weighted)'] = sensitivity_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Sensitivity Score (None, i.e. for each class)'] = sensitivity_score(y_test_max, y_predicted_max,
                                                                                 average=None)

    # =============================================================================
    #    Geometric Mean Score
    # =============================================================================
    metrics['Geometric Mean Score (Normal)'] = geometric_mean_score(y_test_max, y_predicted_max)
    metrics['Geometric Mean Score (Micro)'] = geometric_mean_score(y_test_max, y_predicted_max, average='micro')
    metrics['Geometric Mean Score (Macro)'] = geometric_mean_score(y_test_max, y_predicted_max, average='macro')
    metrics['Geometric Mean Score (Weighted)'] = geometric_mean_score(y_test_max, y_predicted_max, average='weighted')
    if class_specific:
        metrics['Geometric Mean Score (None, i.e. for each class)'] = geometric_mean_score(y_test_max, y_predicted_max,
                                                                                       average=None)

    # =============================================================================
    #    Zero one Loss
    # =============================================================================
    metrics['Zero One Loss'] = zero_one_loss(y_test_max, y_predicted_max)

    # =============================================================================
    #    Make Index Balanced Accuracy with
    # =============================================================================
    # print("\n MIBA with Matthews")
    # geo_mean = make_index_balanced_accuracy(alpha=0.5, squared=True)(hamming_loss)
    # print(geo_mean(y_test_max, y_predicted_max))
    return metrics
