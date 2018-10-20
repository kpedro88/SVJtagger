# get rid of sklearn warnings
from mods import suppress_warn, reset_warn, fprint
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
import time

# restore warnings
reset_warn()

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-i","--input", dest="input", type=str, default="train_uniform_reports.pkl", help="name of .pkl file with reports")
parser.add_argument("-c","--classifiers", dest="classifiers", type=str, default=[], nargs='*', help="plot only for specified classifier(s) (space-separated)")
parser.add_argument("-t","--test", dest="test", type=str, default="", choices=['F','P'], help="suffix for report names (test*, train*)")
parser.add_argument("-s","--suffix", dest="suffix", type=str, default="", help="suffix for plots")
parser.add_argument("-f","--formats", dest="formats", type=str, default=["png"], nargs='*', help="print plots in specified format(s) (space-separated)")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="enable message printing")
args = parser.parse_args()

test = "test"+args.test
train = "train"+args.test
labels = {0: "QCD", 1: "signal"}

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
    fprint("For {:d} signal and {:d} background events the maximum S/sqrt(S+B) is {:.2f} when cutting at {:.2f}".format(ns,nb,signif[imax],bin_edges[imax]))

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

class RepPlot:
    def __init__(self, fun, args=[], kwargs={}):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def create(self):
        start_time = time.time()
        self.plot = self.fun(*self.args,**self.kwargs)
        if args.verbose: fprint("\tCreation time: {:.2f} seconds".format(time.time() - start_time))

    # plot w/ matplotlib because most things not supported for ROOT/TMVA style
    def save(self, name):
        start_time = time.time()
        # separate these
        if name=="CorrelationMatrix":
            for i,iplot in enumerate(self.plot.plots):
                saveplot(name+"_"+labels[i],iplot,figsize=(7,7))
        else:
            saveplot(name,self.plot)
        if args.verbose: fprint("\t  Saving time: {:.2f} seconds".format(time.time() - start_time))

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

# need to reset classifier features to only trained features (eliminating spectators)
def feature_importance(report,columns):
    for name, estimator in report.estimators.items():
        estimator.features = train_features
    return report.feature_importance(grid_columns=columns)

# generate plots
plots = OrderedDict()

plots["CorrelationMatrix"] = RepPlot(reports[test].features_correlation_matrix_by_class,kwargs={'features':train_features, 'labels_dict':labels})
plots["FeatureImportance"] = RepPlot(feature_importance,args=[reports[test],len(report.estimators)])
# learning curves are really slow, disabled for now
#plots["LearningCurveCvM"] = RepPlot(reports[test].learning_curve,args=[KnnBasedCvM(uniform_features, uniform_label=1)])
#plots["LearningCurveRocAuc"] = RepPlot(reports[test].learning_curve,args=[RocAuc()],kwargs={'steps':1})
#plots["LearningCurveSDE"] = RepPlot(reports[test].learning_curve,args=[BinBasedSDE(uniform_features, uniform_label=1)])
plots["PredictionTest"] = RepPlot(reports[train].prediction_pdf,kwargs={'labels_dict':labels, 'bins':50, 'plot_type':'bar'})
plots["PredictionTrain"] = RepPlot(reports[test].prediction_pdf,kwargs={'labels_dict':labels, 'bins':50, 'plot_type':'bar'})
plots["RocCurve"] = RepPlot(reports[test].roc,kwargs={'physics_notion':True})
plots["SpectatorEfficiencies"] = RepPlot(reports[test].efficiencies,kwargs={'features':uniform_features+spectators, 'bins':50, 'labels_dict':labels})
plots["SpectatorProfiles"] = RepPlot(reports[test].profiles,kwargs={'features':uniform_features+spectators, 'bins':50, 'labels_dict':labels, 'grid_columns':len(uniform_features+spectators)})
plots["VariablePdfs"] = RepPlot(reports[test].features_pdf,kwargs={'features':train_features, 'labels_dict':labels, 'bins':50, 'grid_columns':3})

for pname,plot in plots.iteritems():
    if args.verbose: fprint(pname)
    plot.create()
    plot.save(pname)

# this uses the results from plots["PredictionTest"].plot.plot()
plots["MvaEffs"] = RepPlot(mvaeffs,args=[plots["PredictionTest"].plot,labels])
if args.verbose: fprint("MvaEffs")
plots["MvaEffs"].create()
plots["MvaEffs"].save("MvaEffs")
