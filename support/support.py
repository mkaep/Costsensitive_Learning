"""

@author: Martin Käppel
"""

import json
import numpy as np
import os
from support import json_encoder
from sklearn import preprocessing


def create_json(dictionary, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps(dictionary, cls=json_encoder.PythonObjectEncoder))


def export_encoding(path, encoder):
    np.save(os.path.join(path, 'encoding.npy'), encoder.classes_)


def load_encoding(path):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load(os.path.join(path, 'encoding.npy'))
    return encoder
