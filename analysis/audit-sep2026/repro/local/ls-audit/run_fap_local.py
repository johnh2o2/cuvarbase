import importlib.util, sys, types
# load fap_baluev without importing cuvarbase (no pycuda locally)
src = open('/Users/johnhoffman/Documents/cuvarbase/cuvarbase/lombscargle.py').read()
start = src.index('def fap_baluev'); end = src.index('def lomb_scargle_simple')
ns = {}
exec("import numpy as np\nfrom scipy.special import gamma, gammaln\n" + src[start:end], ns)
mod = types.ModuleType('cuvarbase.lombscargle'); mod.fap_baluev = ns['fap_baluev']
sys.modules['cuvarbase'] = types.ModuleType('cuvarbase'); sys.modules['cuvarbase.lombscargle'] = mod
exec(open('exp4_fap.py').read())
