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

