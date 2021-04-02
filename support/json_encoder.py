"""

@author: Martin Käppel
"""
from json import JSONEncoder
import numpy as np


class PythonObjectEncoder(JSONEncoder):

    # Important: The default handler only gets called for values that
    # are not one of the recognised types. So for dict, list, etc. it is not called
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        return JSONEncoder.default(self, obj)
