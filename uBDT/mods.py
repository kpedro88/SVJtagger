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
    from rep.plotting import BarPlot
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
