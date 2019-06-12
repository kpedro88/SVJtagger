from uBDTConfig import uconfig

from E1T import uconfig

del uconfig.training.algorithms["ubdt"]
uconfig.training.weights["flat"] = ["puweight","procweight","flatweightZ30"]
