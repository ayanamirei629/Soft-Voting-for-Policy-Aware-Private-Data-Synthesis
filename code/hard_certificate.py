# Exact hard-boundary certificates for the draft's full product policies.

import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json, time
import numpy as np
import pandas as pd
from data import ROOT, load
import core

def certificate(c, schema):
    seen = {}
    for row in c:
        if schema.name.startswith('xor'):
            key = ()
        elif schema.name == 'adult':
            key = ([0, 1, 1, 1, 2, 2, 3, 3, 3][int(row[3])],)
        else:
            key = ([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3][int(row[3])], int(row[2]) == 3)
        point = tuple(row)
        if key in seen and seen[key] != point:
            return True
        seen[key] = point
    return False
