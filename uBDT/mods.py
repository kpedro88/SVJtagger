# set up path for configs
def config_path(basedir=""):
    import os,sys
    if len(basedir)==0: basedir = os.getcwd()
    sys.path.append(basedir+"/configs")
    sys.path.append(basedir+"/configs/grids")

# this is the only way to get rid of sklearn warnings
def suppress_warn():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn_ = warnings.warn
    warnings.warn = warn

def reset_warn():
    import warnings
    if hasattr(warnings,"warn_"):
        warnings.warn = warnings.warn_
    else:
        print("Already reset warn")

# make status messages useful
def fprint(msg):
    import sys
    print(msg)
    sys.stdout.flush()

def plot_size():
    from rep.plotting import AbstractPlot
    # change default sizing
    orig_init = AbstractPlot.__init__
    def new_init(self):
        orig_init(self)
        self.figsize = (7,7)
    AbstractPlot.__init__ = new_init

def uGB_to_GB(classifier):
    import numpy as np
    classifier.loss_ = classifier.loss
    classifier.loss_.K = 1
    classifier.estimators_ = np.empty((classifier.n_estimators, classifier.loss_.K), dtype=np.object)
    for i,est in enumerate(classifier.estimators):
        classifier.estimators_[i] = est[0]
        classifier.estimators_[i][0].leaf_values = est[1]

def plot_save_histo():
    import numpy
    import matplotlib.pyplot as plt
    from rep.plotting import BarPlot, _COLOR_CYCLE
    def _plot_save_histo(self):
        if not hasattr(self,"histo"): self.histo = {}
        for label, sample in self.data.items():
            color = next(_COLOR_CYCLE)
            prediction, weight, style = sample
            # save the histo in object
            if not label in self.histo:
                if self.value_range is None:
                    c_min, c_max = numpy.min(prediction), numpy.max(prediction)
                else:
                    c_min, c_max = self.value_range

                histo = numpy.histogram(prediction, bins=self.bins, range=(c_min, c_max), weights=weight)

                norm = 1.0
                if self.normalization:
                    norm = float(self.bins) / (c_max - c_min) / sum(weight)

                bin_edges = histo[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
                bin_widths = (bin_edges[1:] - bin_edges[:-1])

                yerr = []
                for i in range(len(bin_edges) - 1):
                    weight_bin = weight[(prediction > bin_edges[i]) * (prediction <= bin_edges[i + 1])]
                    yerr.append(numpy.sqrt(sum(weight_bin * weight_bin)) * norm)

                histo = (numpy.array(histo[0])*norm,histo[1])
                self.histo[label] = (histo,yerr)
            else:
                histo, yerr = self.histo[label]

                bin_edges = histo[1]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
                bin_widths = (bin_edges[1:] - bin_edges[:-1])

            if style == 'filled':
                plt.bar(bin_centers - bin_widths / 2., histo[0], facecolor=color,
                        linewidth=0, width=bin_widths, label=label, alpha=0.5)
            else:
                plt.bar(bin_centers - bin_widths / 2., histo[0],
                        edgecolor=color, color=color, ecolor=color, linewidth=1,
                        width=bin_widths, label=label, alpha=0.5, hatch="/", fill=False)

        # delete underlying data, no longer needed
        self.data = {}

    BarPlot._plot = _plot_save_histo

def fit_separate_weights():
    from rep.metaml import utils
    from rep.metaml.factory import train_estimator, AbstractFactory
    from collections import OrderedDict
    import time

    def fit_weights(self, X, y, sample_weight=None, parallel_profile=None, features=None):
        if features is not None:
            for name, estimator in self.items():
                if estimator.features is not None:
                    print('Overwriting features of estimator ' + name)
                self[name].set_params(features=features)

        # allow specifying different weights for each classifier
        if isinstance(sample_weight,OrderedDict): sample_weight = list(sample_weight.values())
        else: sample_weight = [sample_weight] * len(self)

        start_time = time.time()
        result = utils.map_on_cluster(parallel_profile, train_estimator, list(self.keys()), list(self.values()),
                                      [X] * len(self), [y] * len(self), sample_weight)
        for status, data in result:
            if status == 'success':
                name, estimator, spent_time = data
                self[name] = estimator
                print('model {:12} was trained in {:.2f} seconds'.format(name, spent_time))
            else:
                print('Problem while training on the node, report:\n', data)

        print("Totally spent {:.2f} seconds on training".format(time.time() - start_time))
        return self

    AbstractFactory.fit = fit_weights

def profile_plots():
    import numpy
    from rep import plotting
    from collections import OrderedDict
    from rep.utils import Binner, check_arrays, weighted_quantile
    from rep import utils
    from rep.report import ClassificationReport

    def get_profiles(prediction, spectator, sample_weight=None, bins_number=20, errors=False, ignored_sideband=0.0):
        """
        Construct profile of prediction vs. spectator
        :param binner: Binner object with bins computed from combined sig+bkg spectator value list
        :param prediction: list of probabilities
        :param spectator: list of spectator's values
        :param bins_number: int, count of bins for plot
        :return:
            if errors=False
            tuple (x_values, y_values)
            if errors=True
            tuple (x_values, y_values, y_err, x_err)
            All the parts: x_values, y_values, y_err, x_err are numpy.arrays of the same length.
        """
        prediction, spectator, sample_weight = check_arrays(prediction, spectator, sample_weight)

        spectator_min, spectator_max = weighted_quantile(spectator, [ignored_sideband, (1. - ignored_sideband)])
        mask = (spectator >= spectator_min) & (spectator <= spectator_max)
        spectator = spectator[mask]
        prediction = prediction[mask]
        bins_number = min(bins_number, len(prediction))
        sample_weight = sample_weight if sample_weight is None else numpy.array(sample_weight)[mask]

        binner = Binner(spectator, bins_number=bins_number)
        if sample_weight is None:
            sample_weight = numpy.ones(len(prediction))
        bins_data = binner.split_into_bins(spectator, prediction, sample_weight)

        bin_edges = numpy.array([spectator_min] + list(binner.limits) + [spectator_max])
        x_err = numpy.diff(bin_edges) / 2.
        result = OrderedDict()
        x_values = []
        y_values = []
        N_in_bin = []
        y_err = []
        for num, (masses, probabilities, weights) in enumerate(bins_data):
            y_values.append(numpy.average(probabilities, weights=weights) if len(weights)>0 and sum(weights)>0.0 else 0)
            y_err.append(numpy.sqrt(numpy.cov(probabilities,aweights=numpy.abs(weights),ddof=0)/numpy.sum(weights)) if len(weights)>0 and sum(weights)>0.0 else 0)
            N_in_bin.append(numpy.sum(weights))
            x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)

        x_values, y_values, N_in_bin = check_arrays(x_values, y_values, N_in_bin)
        if errors:
            return (x_values, y_values, y_err, x_err)
        else:
            return (x_values, y_values)

    utils.get_profiles = get_profiles

    def profiles(self, features, mask=None, target_class=1, bins=30, labels_dict=None, ignored_sideband=0.0, errors=False, grid_columns=2):
        """
        Profiles of prediction values for spectators
        :param features: using features (if None then use classifier's spectators)
        :type features: None or list[str]
        :param bins: bins for histogram
        :type bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param target_class: draw probabilities of being classified as target_class
            (default 1, will draw signal probabilities).
            If None, will draw probability corresponding to right class of each event.
        :type target_class: int or None
        :param bool errors: if True then use errorbar, else interpolate function
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: (0, 1) percent of plotting data
        :rtype: plotting.GridPlot
        """
        mask, data, class_labels, weight = self._apply_mask(
            mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        plots = []
        for feature in data.columns:
            for name, prediction in self.prediction.items():
                profiles = OrderedDict()
                prediction = prediction[mask]
                for label, label_name in labels_dict.items():
                    label_mask = class_labels == label
                    target_label = label if target_class is None else target_class
                    profiles[label_name] = utils.get_profiles(
                        prediction[label_mask, target_label],
                        data[feature][label_mask].values,
                        sample_weight=weight[label_mask],
                        bins_number=bins,
                        errors=errors,
                        ignored_sideband=ignored_sideband
                    )

                if errors:
                    plot_fig = plotting.ErrorPlot(profiles)
                else:
                    plot_fig = plotting.FunctionsPlot(profiles)
                plot_fig.xlabel = feature
                plot_fig.ylabel = 'Prediction profile for {}'.format(name)
                plot_fig.ylim = (0, 1)
                plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *plots)
    
    ClassificationReport.profiles = profiles

def eff_target_class():
    from collections import OrderedDict
    from rep import utils
    from rep import plotting
    from rep.report import ClassificationReport

    def efficiencies_target(self, features, thresholds=None, mask=None, target_class=1, bins=30, labels_dict=None, ignored_sideband=0.0, errors=False, grid_columns=2):
        """
        Efficiencies for spectators
        :param features: using features (if None then use classifier's spectators)
        :type features: None or list[str]
        :param bins: bins for histogram
        :type bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param target_class: draw probabilities of being classified as target_class
            (default 1, will draw signal probabilities).
            If None, will draw probability corresponding to right class of each event.
        :type target_class: int or None
        :param list[float] thresholds: thresholds on prediction
        :param bool errors: if True then use errorbar, else interpolate function
        :param labels_dict: label -- name for class label
            if None then {0: 'bck', '1': 'signal'}
        :type labels_dict: None or OrderedDict(int: str)
        :param int grid_columns: count of columns in grid
        :param float ignored_sideband: (0, 1) percent of plotting data
        :rtype: plotting.GridPlot
        """
        mask, data, class_labels, weight = self._apply_mask(
            mask, self._get_features(features), self.target, self.weight)
        labels_dict = self._check_labels(labels_dict, class_labels)

        plots = []
        for feature in data.columns:
            for name, prediction in self.prediction.items():
                prediction = prediction[mask]
                eff = OrderedDict()
                for label, label_name in labels_dict.items():
                    label_mask = class_labels == label
                    target_label = label if target_class is None else target_class
                    eff[label_name] = utils.get_efficiencies(prediction[label_mask, target_label],
                                                             data[feature][label_mask].values,
                                                             bins_number=bins,
                                                             sample_weight=weight[label_mask],
                                                             thresholds=thresholds, errors=errors,
                                                             ignored_sideband=ignored_sideband)

                for label_name, eff_data in eff.items():
                    if errors:
                        plot_fig = plotting.ErrorPlot(eff_data)
                    else:
                        plot_fig = plotting.FunctionsPlot(eff_data)
                    plot_fig.xlabel = feature
                    plot_fig.ylabel = 'Efficiency for {}'.format(name)
                    plot_fig.title = '{} flatness'.format(label_name)
                    plot_fig.ylim = (0, 1)
                    plots.append(plot_fig)

        return plotting.GridPlot(grid_columns, *plots)

    ClassificationReport.efficiencies = efficiencies_target

def plot_2D_text():
    import numpy
    import matplotlib.pyplot as plt

    from rep import utils
    # use pct (-100,100) instead of (-1,1)
    calc_feature_correlation_matrix_old = utils.calc_feature_correlation_matrix
    def calc_feature_correlation_matrix_pct(df,weights=None):
        return 100*calc_feature_correlation_matrix_old(df,weights)

    utils.calc_feature_correlation_matrix = calc_feature_correlation_matrix_pct

    # show value on each point in grid
    from rep.plotting import ColorMap
    def _plot_text(self):
        p = plt.pcolor(self.matrix, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        plt.colorbar(p)
        plt.xlim((0, self.matrix.shape[0]))
        plt.ylim((0, self.matrix.shape[1]))
        for (x,y),z in numpy.ndenumerate(self.matrix):
            plt.text(x+0.5,y+0.5,'{:.0f}'.format(z),ha='center',va='center',bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        if self.labels is not None:
            plt.xticks(numpy.arange(0.5, len(self.labels) + 0.5), self.labels, fontsize=self.fontsize, rotation=90)
            plt.yticks(numpy.arange(0.5, len(self.labels) + 0.5), self.labels, fontsize=self.fontsize)

    ColorMap._plot = _plot_text

def get_eff_safe():
    from rep import utils
    from rep.utils import Binner, check_arrays, weighted_quantile
    import numpy
    from collections import OrderedDict

    def get_efficiencies(prediction, spectator, sample_weight=None, bins_number=20,
                         thresholds=None, errors=False, ignored_sideband=0.0):
        prediction, spectator, sample_weight = \
            check_arrays(prediction, spectator, sample_weight)
    
        spectator_min, spectator_max = weighted_quantile(spectator, [ignored_sideband, (1. - ignored_sideband)])
        mask = (spectator >= spectator_min) & (spectator <= spectator_max)
        spectator = spectator[mask]
        prediction = prediction[mask]
        bins_number = min(bins_number, len(prediction))
        sample_weight = sample_weight if sample_weight is None else numpy.array(sample_weight)[mask]
    
        if thresholds is None:
            thresholds = [weighted_quantile(prediction, quantiles=1 - eff, sample_weight=sample_weight)
                          for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]
    
        binner = Binner(spectator, bins_number=bins_number)
        if sample_weight is None:
            sample_weight = numpy.ones(len(prediction))
        bins_data = binner.split_into_bins(spectator, prediction, sample_weight)
    
        bin_edges = numpy.array([spectator_min] + list(binner.limits) + [spectator_max])
        xerr = numpy.diff(bin_edges) / 2.
        result = OrderedDict()
        for threshold in thresholds:
            x_values = []
            y_values = []
            N_in_bin = []
            for num, (masses, probabilities, weights) in enumerate(bins_data):
                if len(weights)==0 or sum(weights)==0.0: continue
                y_values.append(numpy.average(probabilities > threshold, weights=weights))
                N_in_bin.append(numpy.sum(weights))
                if errors:
                    x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)
                else:
                    x_values.append(numpy.mean(masses))
    
            x_values, y_values, N_in_bin = check_arrays(x_values, y_values, N_in_bin)
            if errors:
                result[threshold] = (x_values, y_values, numpy.sqrt(y_values * (1 - y_values) / N_in_bin), xerr)
            else:
                result[threshold] = (x_values, y_values)
        return result

    utils.get_efficiencies = get_efficiencies

def roc_with_auc():
    from rep.report import ClassificationReport
    from collections import OrderedDict
    import numpy
    from rep import plotting
    from rep import utils
    from sklearn.metrics import auc
    
    def roc(self, mask=None, signal_label=1, physics_notion=False):
        """
        Calculate roc functions for data and return roc plot object
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
        :param bool physics_notion: if set to True, will show signal efficiency vs background rejection,
            otherwise TPR vs FPR.
        :rtype: plotting.FunctionsPlot
        """
        roc_curves = OrderedDict()
        mask, = self._apply_mask(mask)

        classes_labels = set(numpy.unique(self.target[mask]))
        assert len(classes_labels) == 2 and signal_label in classes_labels, \
            'Classes must be 2 instead of {}'.format(classes_labels)

        for name, prediction in self.prediction.items():
            labels_active = numpy.array(self.target[mask] == signal_label, dtype=int)
            (tpr, tnr), _, _ = utils.calc_ROC(prediction[mask, signal_label], labels_active,
                                              sample_weight=self.weight[mask])

            if physics_notion:
                auc_val = auc(tpr, tnr)
                name2 = name+" ({:.3f})".format(auc_val)
                roc_curves[name2] = (tpr, tnr)
                xlabel = 'Signal sensitivity'
                ylabel = 'Bg rejection eff (specificity)'
            else:
                auc_val = auc(1 - tnr, tpr)
                name2 = name+" ({:.3f})".format(auc_val)
                roc_curves[name2] = (1 - tnr, tpr)
                xlabel = 'false positive rate'
                ylabel = 'true positive rate'

        plot_fig = plotting.FunctionsPlot(roc_curves)
        plot_fig.xlabel = xlabel
        plot_fig.ylabel = ylabel
        plot_fig.title = 'ROC curves'
        return plot_fig

    ClassificationReport.roc = roc

def flat_log_loss():
    from hep_ml.losses import AbstractFlatnessLossFunction
    from scipy.special import expit
    import numpy
    
    def negative_gradient_log_loss(self, y_pred):
        y_signed = self.y_signed
        neg_gradient = self._compute_fl_derivatives(y_pred) * self.fl_coefficient
        # adding LogLoss
        neg_gradient += y_signed * self.sample_weight * expit(-y_signed * y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient

    AbstractFlatnessLossFunction.negative_gradient = negative_gradient_log_loss
