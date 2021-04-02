"""

@author: Martin Käppel
"""

import pandas as pd
import numpy as np
import os
from support import support
from sklearn.utils.class_weight import compute_class_weight
import pickle


def calculate_weights(y_true):
    reference_y = np.argmax(y_true, axis=1)
    print("Integers representation: ", reference_y)

    # Calculate with sklearn
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(reference_y), y=reference_y)
    d_class_weights = dict(enumerate(class_weights))

    return d_class_weights
