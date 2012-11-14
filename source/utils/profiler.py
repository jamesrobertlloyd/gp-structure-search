import collections
import functools
import sys
import time

ENABLE_PROFILER = True
TOP_ONLY = True

depth = collections.defaultdict(int)
counts = collections.defaultdict(lambda: collections.defaultdict(int))
total_time = collections.defaultdict(lambda: collections.defaultdict(float))

def reset(category=None):
    global counts, total_time
    if category is None:
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
        total_time = collections.defaultdict(lambda: collections.defaultdict(float))
    else:
        counts[category] = collections.defaultdict(int)
        total_time[category] = collections.defaultdict(float)

def get_key(name, args):
    k = []
    for arg in args:
        if hasattr(arg, 'shape_str'):
            k.append((str(arg.__class__), arg.shape_str))
        elif hasattr(arg, 'shape'):
            k.append((str(arg.__class__), arg.shape))
    return (name,) + tuple(k)


class profiled:
    def __init__(self, category):
        self.category = category

    def __call__(self, fn):
        if not ENABLE_PROFILER:
            return fn

        name = fn.__name__

        @functools.wraps(fn)
        def profiled_fn(*args, **kwargs):
            global depth
            t0 = time.clock()
            depth[self.category] += 1
            ans = fn(*args, **kwargs)
            depth[self.category] -= 1
            if depth[self.category] == 0 or not TOP_ONLY:
                key = get_key(name, args)
                counts[self.category][key] += 1
                total_time[self.category][key] += time.clock() - t0
            return ans

        return profiled_fn


def summarize(category, cutoff=0.5, outstr=sys.stdout):
    tt = total_time[category]
    c = counts[category]
    srtd = sorted(tt.keys(), key=lambda k: tt[k], reverse=True)
    for k in srtd:
        if tt[k] < cutoff:
            continue
        print >> outstr, '%1.2f seconds for %d calls' % (tt[k], c[k])
        print >> outstr, k[0]
        for tp, sz in k[1:]:
            print >> outstr, '    %s %s' % (tp, sz)
        print >> outstr


    
