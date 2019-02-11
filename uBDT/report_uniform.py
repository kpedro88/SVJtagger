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
import time, os

# restore warnings
reset_warn()

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d","--dir", dest="dir", type=str, default="", help="directory for train_uniform_reports.pkl input file (required)")
parser.add_argument("-o","--outdir", dest="outdir", type=str, default="", help="directory for output pngs (if different from input dir)")
parser.add_argument("-C","--config", dest="config", type=str, default="test1", help="config to provide parameters")
parser.add_argument("-c","--classifiers", dest="classifiers", type=str, default=[], nargs='*', help="plot only for specified classifier(s) (space-separated)")
parser.add_argument("-t","--test", dest="test", type=str, default="", choices=['flat','proc'], help="suffix for report names (test*, train*)")
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

# fix bkg efficiency plots
from mods import eff_target_class
eff_target_class()
from mods import get_eff_safe
get_eff_safe()

# show text on 2D plots
from mods import plot_2D_text
plot_2D_text()

# show AUC for ROC curves
from mods import roc_with_auc
roc_with_auc()

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
            tests[getlabel(label,labels)][dataset] = histo[0]
    # conduct tests
    from scipy import stats
    for label, test in tests.iteritems():
        ks, pv = stats.ks_2samp(test["train"],test["test"])
        test["result"] = pv
    # update labels and combine into one plot
    ndata = OrderedDict()
    for dataset,barplot in sorted(barplots.iteritems()):
        for label,data in sorted(barplot.data.iteritems()):
            nlabel = label+" ("+dataset+")"
            if dataset=="test": nlabel += " (ks = "+"{:.3f}".format(tests[getlabel(label,labels)]["result"])+")"
            ndata[nlabel] = (data[0], data[1], 'not_filled' if dataset=="train" else 'filled')
    # plot together
    plot_fig = plotting.BarPlot(ndata,bins=barplots["test"].bins,normalization=barplots["test"].normalization,value_range=barplots["test"].value_range)
    plot_fig.xlabel = barplots["test"].xlabel
    plot_fig.ylabel = barplots["test"].ylabel
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

def saveplots(plots,config="",verbose=False):
    for pname,plot in plots.iteritems():
        if verbose: fprint(pname)
        plot.create(config)
        plot.save(args.outdir+"/"+pname)

class RepPlot:
    def __init__(self, fun, args=[], kwargs={}):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def create(self,prefix=""):
        start_time = time.time()
        self.plot = self.fun(*self.args,**self.kwargs)
        if len(prefix)>0:
            if isinstance(self.plot,plotting.GridPlot):
                for p in self.plot.plots:
                    p.title = prefix+" "+p.title
            else:
                self.plot.title = prefix+" "+self.plot.title
        if args.verbose: fprint("\tCreation time: {:.2f} seconds".format(time.time() - start_time))

    # plot w/ matplotlib because most things not supported for ROOT/TMVA style
    def save(self, name):
        start_time = time.time()
        # separate these
        if "CorrelationMatrix" in name:
            for i,iplot in enumerate(self.plot.plots):
                saveplot(name+"_"+labels[i],iplot,figsize=(7,7))
        else:
            saveplot(name,self.plot)
        if args.verbose: fprint("\t  Saving time: {:.2f} seconds".format(time.time() - start_time))

# get reports
if args.verbose: fprint("Start loading input")
with open(args.dir+"/train_uniform_reports.pkl",'rb') as infile:
    reports = pickle.load(infile)
if args.verbose: fprint("Finish loading input")

from mods import config_path
config_path() 
args.config = args.config.replace(".py","")
uconfig = getattr(__import__(args.config,fromlist="uconfig"),"uconfig")

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

# need to reset classifier features to only trained features (eliminating spectators)
def feature_importance(report,columns):
    for name, estimator in report.estimators.items():
        estimator.features = uconfig.features.train
    return report.feature_importance(grid_columns=columns)

# get rid of last QCD bin in profiles
def profiles(report,**kwargs):
    plot = report.profiles(**kwargs)
    for p in plot.plots:
        p.functions['QCD'] = (p.functions['QCD'][0][:-1],p.functions['QCD'][1][:-1])
    return plot

# generate plots
plots = OrderedDict()

plots["CorrelationMatrix"] = RepPlot(reports[test].features_correlation_matrix_by_class,kwargs={'features':uconfig.features.train, 'labels_dict':labels, 'vmin':-100, 'vmax':100})
plots["FeatureImportance"] = RepPlot(feature_importance,args=[reports[test],len(report.estimators)])
# learning curves are really slow, disabled for now
#plots["LearningCurveCvM"] = RepPlot(reports[test].learning_curve,args=[KnnBasedCvM(uconfig.features.uniform, uniform_label=1)])
#plots["LearningCurveRocAuc"] = RepPlot(reports[test].learning_curve,args=[RocAuc()],kwargs={'steps':1})
#plots["LearningCurveSDE"] = RepPlot(reports[test].learning_curve,args=[BinBasedSDE(uconfig.features.uniform, uniform_label=1)])
plots["PredictionTest"] = RepPlot(reports[train].prediction_pdf,kwargs={'labels_dict':labels, 'bins':50, 'plot_type':'bar'})
plots["PredictionTrain"] = RepPlot(reports[test].prediction_pdf,kwargs={'labels_dict':labels, 'bins':50, 'plot_type':'bar'})
plots["RocCurve"] = RepPlot(reports[test].roc,kwargs={'physics_notion':True})
plots["SpectatorEfficiencies"] = RepPlot(reports[test].efficiencies,kwargs={'features':uconfig.features.uniform+uconfig.features.spectator, 'bins':50, 'labels_dict':labels})
plots["SpectatorProfiles"] = RepPlot(profiles,args=[reports[test]],kwargs={'features':uconfig.features.uniform+uconfig.features.spectator, 'bins':50, 'labels_dict':labels, 'grid_columns':len(uconfig.features.uniform+uconfig.features.spectator)})
plots["VariablePdfs"] = RepPlot(reports[test].features_pdf,kwargs={'features':uconfig.features.train, 'labels_dict':labels, 'bins':50, 'grid_columns':3})

saveplots(plots,"["+args.config+"]",args.verbose)

# "derived" plots
plots2 = OrderedDict()

# this uses the results from plots["PredictionTest"].plot.plot()
plots2["MvaEffs"] = RepPlot(mvaeffs,args=[plots["PredictionTest"].plot,labels])

# this uses the results from both prediction plots
preds = {
    "train": plots["PredictionTrain"].plot,
    "test": plots["PredictionTest"].plot
}
plots2["OverTrain"] = RepPlot(kstest,args=[preds,labels])

saveplots(plots2,"["+args.config+"]",args.verbose)
