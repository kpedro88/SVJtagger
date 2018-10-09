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

Output files produced:
* `train_uniform_classifiers.pkl`: classifiers in python pickle format
* `train_uniform_reports.pkl`: report in python pickle format
* `TMVA_GradBoost_weights.xml`: BDT in TMVA format
* `TMVA_uGBFL_weights.xml`: uBDT in TMVA format

## Producing plots

```
python report_uniform.py
```

This uses the saved `train_uniform_reports.pkl` output file.
