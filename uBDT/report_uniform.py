# get rid of sklearn warnings
from mods import suppress_warn, reset_warn
suppress_warn()

from rep.estimators import SklearnClassifier
from hep_ml import uboost, gradientboosting as ugb, losses
from rep.metaml import ClassifiersFactory
from rep.report.metrics import RocAuc
from rep import plotting
from hep_ml.metrics import BinBasedSDE, KnnBasedCvM
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict

# restore warnings
reset_warn()

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-i","--input", dest="input", type=str, default="train_uniform_reports.pkl", help="name of .pkl file with reports")
parser.add_argument("-c","--classifiers", dest="classifiers", type=str, default=[], nargs='*', help="plot only for specified classifier(s) (space-separated)")
parser.add_argument("-s","--suffix", dest="suffix", type=str, default="", help="suffix for plots")
parser.add_argument("-f","--formats", dest="formats", type=str, default=["png"], nargs='*', help="print plots in specified format(s) (space-separated)")
args = parser.parse_args()

# change default sizing
from mods import plot_size
plot_size()

# keep histos for later use
from mods import plot_save_histo
plot_save_histo()

# enable profile plots
from mods import profile_plots
profile_plots()

# make sig vs bkg eff
def mvaeffs(barplot,labels):
    effs = {}
    # for s/sqrt(s+b) calc
    nb = 1000
    ns = 1000
    bin_edges = None
    # get efficiencies
    for label, histo in barplot.histo.iteritems():
        sb = ""
        if labels[0] in label: sb = "B"
        elif labels[1] in label: sb = "S"
        bin_edges = histo[1]
        norm = np.sum(histo[0],dtype=float)
        effs[sb] = np.flip(np.cumsum(np.flip(histo[0]),dtype=float))/norm
    
    # find max s/sqrt(s+b)
    signif = effs["S"]*ns/np.sqrt(effs["S"]*ns+effs["B"]*nb)
    imax = np.argmax(signif)
    print("For {:d} signal and {:d} background events the maximum S/sqrt(S+B) is {:.2f} when cutting at {:.2f}".format(ns,nb,signif[imax],bin_edges[imax]))

    # make plot
    eff_curves = OrderedDict()
    eff_curves[labels[0]] = (bin_edges[:-1],effs["B"])
    eff_curves[labels[1]] = (bin_edges[:-1],effs["S"])
    plot_fig = plotting.FunctionsPlot(eff_curves)
    plot_fig.xlabel = "prediction"
    plot_fig.ylabel = "efficiency"
    return plot_fig

def saveplot(pname,plot,figsize=None):
    plot.plot(new_plot=True,figsize=figsize)
    fig = plt.gcf()
    fname = pname
    if len(args.suffix)>0: fname += "_"+args.suffix
    for format in args.formats:
        fargs = {}
        if format=="png": fargs = {"dpi":100}
        elif format=="pdf": fargs = {"bbox_inches":"tight"}
        fig.savefig(fname+"."+format,**fargs)

# get reports
with open(args.input,'rb') as infile:
    reports = pickle.load(infile)

# check for subset of classifiers
if len(args.classifiers)>0:
    for rname,report in reports.iteritems():
        est_old = report.estimators
        pred_old = report.prediction
        est_new = ClassifiersFactory()
        pred_new = OrderedDict()
        for classifier in args.classifiers:
            if classifier in est_old:
                est_new[classifier] = est_old[classifier]
                pred_new[classifier] = pred_old[classifier]
            else:
                raise ValueError("Requested classifier "+classifier+" not found in report "+rname)
        report.estimators = est_new
        report.prediction = pred_new

# provides: uniform_features, train_features, spectators, all_vars
from features import *

# generate plots
labels = {0: "QCD", 1: "signal"}
plots = {}

plots["SpectatorEfficiencies"] = reports["test"].efficiencies(features=uniform_features+spectators, bins=50, labels_dict=labels)
plots["SpectatorProfiles"] = reports["test"].profiles(features=uniform_features+spectators, bins=50, labels_dict=labels, grid_columns=len(uniform_features+spectators))
plots["CorrelationMatrix"] = reports["test"].features_correlation_matrix_by_class(features=train_features, labels_dict=labels)
plots["VariablePdfs"] = reports["test"].features_pdf(features=train_features, labels_dict=labels, bins=50, grid_columns=3)
plots["LearningCurveRocAuc"] = reports["test"].learning_curve(RocAuc(), steps=1)
plots["LearningCurveSDE"] = reports["test"].learning_curve(BinBasedSDE(uniform_features, uniform_label=1))
plots["LearningCurveCvM"] = reports["test"].learning_curve(KnnBasedCvM(uniform_features, uniform_label=1))
plots["PredictionTrain"] = reports["test"].prediction_pdf(labels_dict=labels, bins=50, plot_type='bar')
plots["PredictionTest"] = reports["train"].prediction_pdf(labels_dict=labels, bins=50, plot_type='bar')
plots["RocCurve"] = reports["test"].roc(physics_notion=True)

# need to reset classifier features to only trained features (eliminating spectators)
for name, estimator in reports["test"].estimators.items():
    estimator.features = train_features
plots["FeatureImportance"] = reports["test"].feature_importance(grid_columns=len(report.estimators))

# plot w/ matplotlib because most things not supported for ROOT/TMVA style
for pname,plot in sorted(plots.iteritems()):
    # separate these
    if pname=="CorrelationMatrix":
        for i,iplot in enumerate(plot.plots):
            saveplot(pname+"_"+labels[i],iplot,figsize=(7,7))
    else:
        saveplot(pname,plot)

# this uses the results from plots["PredictionTest"].plot()
plots["MvaEffs"] = mvaeffs(plots["PredictionTest"],labels)
saveplot("MvaEffs",plots["MvaEffs"])
