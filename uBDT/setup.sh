#!/bin/bash

# get CMSSW
export SCRAM_ARCH=slc6_amd64_gcc700
CMSSWVER=CMSSW_10_3_0_pre5
scram project ${CMSSWVER}
cd ${CMSSWVER}/src/
# cmsenv
eval `scramv1 runtime -sh`

# get tagger repo
git clone git@github.com:kpedro88/SVJtagger.git
cd SVJtagger/uBDT

# get sklearn -> TMVA converter
git clone git@github.com:yuraic/koza4ok.git
ln -s koza4ok/skTMVA .
