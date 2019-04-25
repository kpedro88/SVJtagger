#!/bin/bash

# get CMSSW
export SCRAM_ARCH=slc6_amd64_gcc700
CMSSWVER=CMSSW_10_3_0
scram project ${CMSSWVER}
cd ${CMSSWVER}/src/
# cmsenv
eval `scramv1 runtime -sh`

# get batch code
git clone git@github.com:kpedro88/CondorProduction.git Condor/Production
scram b -j 8

# get tagger repo
git clone git@github.com:kpedro88/SVJtagger.git
cd SVJtagger/uBDT

# get sklearn -> TMVA converter
git clone git@github.com:kpedro88/koza4ok.git
ln -s koza4ok/skTMVA .

# extra batch setup
cd batch
python $CMSSW_BASE/src/Condor/Production/python/linkScripts.py
