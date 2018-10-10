# this is the only way to get rid of sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warn_ = warnings.warn

def suppress_warn():
    warnings.warn = warn

def reset_warn():
    warnings.warn = warn_

def plot_size():
    from rep.plotting import AbstractPlot
    # change default sizing
    orig_init = AbstractPlot.__init__
    def new_init(self):
        orig_init(self)
        self.figsize = (7,7)
    AbstractPlot.__init__ = new_init

