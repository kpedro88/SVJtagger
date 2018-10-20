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
        self.histo = {}
        for label, sample in self.data.items():
            color = next(_COLOR_CYCLE)
            prediction, weight, style = sample
            if self.value_range is None:
                c_min, c_max = numpy.min(prediction), numpy.max(prediction)
            else:
                c_min, c_max = self.value_range
            histo = numpy.histogram(prediction, bins=self.bins, range=(c_min, c_max), weights=weight)
            # save the histo in object
            self.histo[label] = histo
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

            if style == 'filled':
                plt.bar(bin_centers - bin_widths / 2., numpy.array(histo[0]) * norm, facecolor=color,
                        linewidth=0, width=bin_widths, label=label, alpha=0.5)
            else:
                plt.bar(bin_centers - bin_widths / 2., norm * numpy.array(histo[0]),
                        edgecolor=color, color=color, ecolor=color, linewidth=1,
                        width=bin_widths, label=label, alpha=0.5, hatch="/", fill=False)

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
            y_values.append(numpy.average(probabilities, weights=weights) if len(weights)>0 else 0)
            y_err.append(numpy.sqrt(numpy.cov(probabilities,aweights=weights,ddof=0)/numpy.sum(weights)) if len(weights)>0 else 0)
            N_in_bin.append(numpy.sum(weights))
            x_values.append((bin_edges[num + 1] + bin_edges[num]) / 2.)

        x_values, y_values, N_in_bin = check_arrays(x_values, y_values, N_in_bin)
        if errors:
            return (x_values, y_values, y_err, x_err)
        else:
            return (x_values, y_values)

    utils.get_profiles = get_profiles

    def profiles(self, features, mask=None, bins=30, labels_dict=None, ignored_sideband=0.0, errors=False, grid_columns=2):
        """
        Efficiencies for spectators
        :param features: using features (if None then use classifier's spectators)
        :type features: None or list[str]
        :param bins: bins for histogram
        :type bins: int or array-like
        :param mask: mask for data, which will be used
        :type mask: None or numbers.Number or array-like or str or function(pandas.DataFrame)
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
                    profiles[label_name] = utils.get_profiles(
                        prediction[label_mask, label],
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

