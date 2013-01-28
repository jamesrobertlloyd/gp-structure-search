import os

import base
import config
import mit_job_controller as mjc

class Scheduler:
    def evaluate_kernels(self, kernels, X, y):
        scored_kernels = []
        for i, k in enumerate(kernels):
            print 'Evaluating %d of %d...' % (i+1, len(kernels))
            sk = mjc.evaluate_kernel(k, X, y)
            scored_kernels.append(sk)
        return scored_kernels

def run(data_name, max_depth=3, params=None):
    if not os.path.exists(config.TEMP_PATH):
        os.mkdir(config.TEMP_PATH)
    X, y = base.load_data(data_name)
    scheduler = Scheduler()
    if params is None:
        params = base.SearchParams.default()
    base.perform_search(X, y, scheduler, max_depth, params, verbose=True)



