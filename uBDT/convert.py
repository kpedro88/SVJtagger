import sys
import cPickle as pickle
from skTMVA import skTMVA

pkl = sys.argv[1]
prefix = pkl.split('_')[0]

with open(pkl,'rb') as infile:
    classifiers = pickle.load(infile)

wname_bdt = prefix+"_TMVA_bdt_weights.xml"
wname_ubdt = prefix+"_TMVA_ubdt_weights.xml"

# save in TMVA format
tmva_vars = []

if "bdt" in classifiers:
    if len(tmva_vars)==0:
        tmva_vars = [(f, 'F') for f in classifiers["bdt"].features]
    skTMVA.convert_bdt__Grad(classifiers["bdt"],tmva_vars,wname_bdt)

if "ubdt" in classifiers:
    if len(tmva_vars)==0:
        tmva_vars = [(f, 'F') for f in classifiers["ubdt"].clf.train_features]
    from mods import uGB_to_GB
    uGB_to_GB(classifiers["ubdt"])
    skTMVA.convert_bdt__Grad(classifiers["ubdt"],tmva_vars,wname_ubdt)
