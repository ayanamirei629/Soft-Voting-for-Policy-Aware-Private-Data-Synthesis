# Sampling intervention with the hybrid reference's candidate-count schedule.

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import crossover_study as study
from data import load
from run import identity, atomic_json

def matched_schedule_schema(schema):
    result = copy.copy(schema)
    original = schema.mutate
    calls = 0

    def mutate(points, rate, rng):
        nonlocal calls
        mutated = original(points, rate, rng)
        output = mutated if calls < 2 else np.vstack([points, mutated])
        calls += 1
        return output
    result.mutate = mutate
    return result

def worker(cfg):
    key = (cfg['dataset'], cfg['data_seed'])
    if key not in study.DATA:
        study.DATA[key] = load(cfg['dataset'], 0, cfg['data_seed'])
    original = study.DATA[key]
    study.DATA[key] = (matched_schedule_schema(original[0]),) + original[1:]
    try:
        result = study.worker(cfg)
        if 'error' not in result:
            path = study.HOME / 'results/sampling' / f'{identity(cfg)}.json'
            record = json.loads(path.read_text())
            assert [h['m_candidates'] for h in record['history']] == [512, 512, 512, 1024, 1024, 1024, 1024, 1024]
            record['selection_intervention'] = 'Sampling only; candidate counts, parent/mutant rule and rates matched to hybrid reference.'
            atomic_json(path, record)
        return result
    finally:
        study.DATA[key] = original
