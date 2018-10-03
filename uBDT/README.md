# SVJ tagger: uBDT

This is a uniform gradient-boosted BDT using flattening loss from hep_ml.

## Setup

```
wget https://raw.githubusercontent.com/kpedro88/SVJtagger/master/uBDT/setup.sh
chmod +x setup.sh
./setup.sh
cd CMSSW_10_3_0_pre5/src/SVJtagger/uBDT
cmsenv
```

## Running

```
python train_uniform.py
```

For rapid prototyping, the training dataset is specified as 1% of the input data.
For physics performance tests, the training dataset should be increased to 50% of the input data.
