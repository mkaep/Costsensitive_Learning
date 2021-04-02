"""
Created on 18.02.2021

@author: Martin Kaeppel

Generalized the source code of Pasquadibisceglie; hereby the get_image method was generalized to
support all events logs not only Heldpesk and BPIC12.
"""

import numpy as np


def get_image(act_val, time_val, max_trace_length, n_activity, activities):
    i = 0
    matrix_zero = [max_trace_length, n_activity, 2]
    image = np.zeros(matrix_zero, dtype=np.float16)
    list_image = []
    # Iteriere ueber traces
    while i < len(time_val):
        j = 0
        list_act = []
        list_temp = []
        dictionary_cont = dict()
        for k in activities:
            dictionary_cont.update({k : 0})
        dictionary_diff = dict()
        for k in activities:
            dictionary_diff.update({k : 0})

        while j < (len(act_val.iat[i, 0]) - 1):
            start_trace = time_val.iat[i, 0][0]
            dictionary_cont[act_val.iat[i, 0][0 + j]] = dictionary_cont[act_val.iat[i, 0][0 + j]] + 1
            dictionary_diff[act_val.iat[i, 0][0 + j]] = time_val.iat[i, 0][0 + j] - start_trace

            temp_cond_list = []
            for key in dictionary_cont:
                temp_cond_list.append(dictionary_cont[key])
            list_act.append(temp_cond_list)

            temp_diff_list = []
            for key in dictionary_diff:
                temp_diff_list.append(dictionary_diff[key])
            list_temp.append(temp_diff_list)
            j = j + 1
            cont = 0
            lenk = len(list_act) - 1
            while cont <= lenk:
                for l in range(0, n_activity):
                    image[(max_trace_length - 1) - cont][l] = [list_act[lenk - cont][l], list_temp[lenk - cont][l]]
                cont = cont + 1

            if cont == 1:
                pass
            else:
                list_image.append(image)
                image = np.zeros(matrix_zero, dtype=np.float16)
        i = i + 1
    return list_image


def get_label(act):
    i = 0
    list_label = []
    # Iteriere ueber die Case IDs
    while i < len(act):
        j = 0
        while j < (len(act.iat[i, 0]) - 1):
            if j > 0:
                list_label.append(act.iat[i, 0][j + 1])
            else:
                pass
            j = j + 1
        i = i + 1
    return list_label
