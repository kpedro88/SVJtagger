import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# get rid of sklearn warnings
from mods import suppress_warn, reset_warn, fprint
suppress_warn()
from mods import config_path
config_path()

from rep.estimators import SklearnClassifier
from hep_ml import uboost, gradientboosting as ugb, losses
from rep.metaml import ClassifiersFactory
from rep.report.metrics import RocAuc
from rep import plotting
from hep_ml.metrics import BinBasedSDE, KnnBasedCvM
import numpy as np
import pandas as pd
import cPickle as pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import time, os

# restore warnings
reset_warn()

def rocauc(rocplot,classifier):
    from sklearn.metrics import auc
    for name, data_xy in rocplot.functions.iteritems():
        if classifier in name:
            x_val, y_val = data_xy
            return auc(x_val,y_val)
    return 0.

# make sig vs bkg eff
def mvaeffs(effplot,labels,wps=[0.1]):
    effs = {
        "B": effplot.functions[labels[0]][1],
        "S": effplot.functions[labels[1]][1],
    }
    bin_edges = effplot.functions[labels[0]][0]

    # find different working points
    wps_cut = []
    wps_eff_bkg = []
    wps_eff_sig = []
    for wp in wps:
        idx = (np.abs(effs["B"] - wp)).argmin()
        wps_cut.append(bin_edges[idx])
        wps_eff_bkg.append(effs["B"][idx])
        wps_eff_sig.append(effs["S"][idx])

    return (wps_cut,wps_eff_bkg,wps_eff_sig)

def getlabel(label,labels):
    rlabel = ""
    if labels[0] in label: rlabel = labels[0]
    elif labels[1] in label: rlabel = labels[1]
    return rlabel

def kstest(barplots,labels):
    tests = {}
    for li,label in labels.iteritems():
        tests[label] = {"result": 0}
    for dataset,barplot in barplots.iteritems():
        for label,histo in barplot.histo.iteritems():
            tests[getlabel(label,labels)][dataset] = histo[0][0]
    # conduct tests
    from scipy import stats
    for label, test in tests.iteritems():
        ks, pv = stats.ks_2samp(test["train"],test["test"])
        test["result"] = pv
    return tests

def flatness(gridplot,label,varname,index=0):
    for plot in gridplot.plots:
        if label in plot.title and varname in plot.xlabel:
            x_val, y_val = plot.functions.values()[index]
            corr = np.corrcoef(x_val,y_val)
            return abs(corr[0][1])
    return 0.

def saveplot(pname,plot,figsize=None):
    curve = plot.plot(new_plot=True,figsize=figsize)
    fig = plt.gcf()
    fname = args.outdir+"/"+pname
    if len(args.suffix)>0: fname += "_"+args.suffix
    for format in args.formats:
        fargs = {}
        if format=="png": fargs = {"dpi":100}
        elif format=="pdf": fargs = {"bbox_inches":"tight"}
        fig.savefig(fname+"."+format,**fargs)

if __name__=="__main__":
    # check arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--dir", dest="dir", type=str, default="", help="directory for repplots pkl input file (required)")
    parser.add_argument("-o","--outdir", dest="outdir", type=str, default="", help="directory for output (if different from input dir)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-C","--config", dest="config", type=str, default="", help="configs to compare")
    group.add_argument("-G","--grid", dest="grid", type=int, default=-1, help="grid version")
    parser.add_argument("-c","--classifier", dest="classifier", type=str, default="", help="plot only for specified classifier")
    parser.add_argument("-s","--suffix", dest="suffix", type=str, default="", help="suffix for plots")
    parser.add_argument("-f","--formats", dest="formats", type=str, default=["png"], nargs='*', help="print plots in specified format(s) (space-separated)")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="enable message printing")
    args = parser.parse_args()

    if len(args.dir)==0:
        parser.error("Required argument: --dir")
    if len(args.outdir)==0:
        args.outdir = args.dir
    elif not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    labels = {0: "QCD", 1: "signal"}
    configs = []
    is_grid = False
    if args.grid>0:
        from makeGrid import getPointName, makeGridPoints
        configs = makeGridPoints(args.grid)
        is_grid = True
    else:
        configs = args.config.split(",")

    data_dict = {
        "name": [],
        "auc": [],
        "sigeff": [],
        "sigovertrain": [],
        "bkgovertrain": [],
        "bkgflatness": [],
    }
    # add extra params for grid
    if is_grid:
        data_dict["number"] = []
        from makeGrid import getVars
        gridvars = getVars(args.grid)
        for gridvar in gridvars:
            data_dict[gridvar] = []

    # populate data dict from plots
    for ic,config in enumerate(configs):
        configname = config
        if is_grid: configname = getPointName(ic,args.config,args.grid)
        with open(args.dir+"/"+configname+"/repplots_"+args.suffix+".pkl",'rb') as infile:
            repplots = pickle.load(infile)
        data_dict["name"].append(configname)
        # grid-specific data
        if is_grid:
            data_dict["number"].append(ic)
            for gridvar,gridval in zip(gridvars,config):
                data_dict[gridvar].append(gridval)
        # data derived from plots
        data_dict["auc"].append(rocauc(repplots["RocCurve"],args.classifier))
        data_dict["sigeff"].append(mvaeffs(repplots["MvaEffs"],labels)[2][0])
        overtrain = kstest({"train": repplots["PredictionTrain"], "test": repplots["PredictionTest"]},labels)
        data_dict["sigovertrain"].append(overtrain["signal"]["result"])
        data_dict["bkgovertrain"].append(overtrain["QCD"]["result"])
        data_dict["bkgflatness"].append(flatness(repplots["SpectatorEfficiencies"],"QCD","mt"))

    # convert dict to pandas df
    df = pd.DataFrame.from_dict(data_dict)
    print df
