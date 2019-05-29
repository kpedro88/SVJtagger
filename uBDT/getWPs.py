import uproot as up
import numpy as np
from mods import fprint

import sys
fname = sys.argv[1]

f = up.open(fname)
bkg = sys.argv[2]

bkg_key = next(k for k in f.allkeys() if bkg in k)

h = f[bkg_key].numpy

bin_edges = h[1]
bin_vals = h[0]

norm = np.sum(bin_vals,dtype=float)
effs = np.flip(np.cumsum(np.flip(bin_vals),dtype=float))/norm

# find different working points
wps = [0.2, 0.1, 0.05, 0.01]
for wp in wps:
    idx = (np.abs(effs - wp)).argmin()
    fprint("{:.2f} working point: discr = {:.2f} (bkg eff = {:.2f})".format(wp,bin_edges[idx],effs[idx]))
