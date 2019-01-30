from collections import Mapping

# combination of https://stackoverflow.com/a/15247892 and https://stackoverflow.com/a/18348004
class ConfigBase(object):
    def __init__(self, classtype):
        self._type = classtype
    # todo: add repr
def ConfigFactory(name, fields, defaults=()):
    def __init__(self, **kwargs):
        self._fields = fields
        for index,field in enumerate(self._fields):
            val = None
            if field in kwargs: val = kwargs[field]
            elif isinstance(defaults,Mapping) and field in defaults: val = defaults[field]
            elif index<len(defaults): val = defaults[index]
            setattr(self,field,val)
        ConfigBase.__init__(self,name)
        self.initialized = True
    def __setattr__(self, name, value):
        if hasattr(self,"initialized") and self.initialized and name not in self._fields:
            raise Exception("invalid argument")
        ConfigBase.__setattr__(self,name,value)
    newconfig = type(name, (ConfigBase,), {"__init__": __init__})
    newconfig.__setattr__ = __setattr__
    return newconfig

uDataset = ConfigFactory('uDataset',['path','signal','background'])
uTraining = ConfigFactory('uTraining',['size','weights','algorithms'])
uHyper = ConfigFactory('uHyper',['max_depth','n_estimators','subsample','learning_rate','min_samples_leaf','fl_coefficient','power','uniform_label','n_bins'])

class uFeatures(ConfigFactory('uFeatures',['uniform','train','spectator'])):
    def all_vars(self):
        tmp_vars = []
        for v in self._fields:
            if isinstance(getattr(self,v),list): tmp_vars.extend(getattr(self,v))
            else: tmp_vars.append(getattr(self,v))
        return tmp_vars
    
uBDTConfig = ConfigFactory('uBDTConfig',['dataset','features','training','hyper'],(uDataset(),uFeatures(),uTraining(),uHyper()))

uconfig = uBDTConfig()
