import pandas as pd
import numpy as np
# from utils import plotting

def load_reference_log(path):
    reference_log_df = pd.read_csv(path, sep=",")
    return reference_log_df

def prescriptive_analysis(reference_log):
    expected_colums = {'CaseID', 'Activity', 'Timestamp'}

    if set(reference_log.columns) == expected_colums:
        n_caseid = reference_log['CaseID'].nunique()
        activities = set(reference_log['Activity'])
        n_activities = len(activities)

        print("Number of CaseIDs: ", n_caseid)
        print("Activities: ", activities)
        print("Number of Unique Activities: ", n_activities)
        print("Number of Events: ", reference_log['CaseID'].count())

        cont_trace = reference_log['CaseID'].value_counts(dropna=False)
        max_trace_length = max(cont_trace)
        min_trace_length = min(cont_trace)
        avg_trace_length = np.mean(cont_trace)

        print("Maximum trace length: ", max_trace_length)
        print("Minimum trace length: ", min_trace_length)
        print("Mean trace length: ", avg_trace_length)
        return max_trace_length, n_caseid, n_activities, activities
    else:
        print("Error: The input format does not match!")
        return


def get_label_distribution(y, activities):
    labels_counted = dict()
    for act in activities:
        labels_counted.update({act: y.count(act)})
    return labels_counted
    #plotting.plot_barchart_from_dictionary(labels_counted, "Label Distribution (" + title +")", "Label", "Number of Occurence")


def get_activity_distribution(log, activities):
    # TO DO pruefen, ob der log die richtigen columsn aht
    # Count the activities
    activities_counted = dict()
    for act in activities:
        activities_counted.update(({act: (log['Activity'] == act).sum()}))
    return activities_counted
    #plotting.plot_barchart_from_dictionary(activities_counted, "Activity Distribution (" + title + ")", "Activities", "Number of Occurence")





