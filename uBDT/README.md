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

Arguments:
* `-t, --train-test-size [size]`: size for test and train datasets (default = 0.5)
* `-s, --suffix [suffix]`: suffix for output files
* `-v, --verbose`: enable message printing (default = False)

For rapid prototyping, use `-t 0.01`.

Output files produced:
* `train_uniform_classifiers[_suffix].pkl`: classifiers in python pickle format
* `train_uniform_reports[_suffix].pkl`: report in python pickle format
* `TMVA_GradBoost_weights[_suffix].xml`: BDT in TMVA format
* `TMVA_uGBFL_weights[_suffix].xml`: uBDT in TMVA format

## Producing plots

```
python report_uniform.py
```

Arguments:
* `-i, --input [file]`: name of .pkl file with reports (default = train_uniform_reports.pkl)
* `-c, --classifiers [list]`: plot only for specified classifier(s) (space-separated) (default = [] -> all)
* `-t, --test {F,P}`: suffix for report names (test*, train*) (F = flat weight, P = proc weight)
* `-s, --suffix [suffix]`: suffix for plots
* `-f, --formats [list]`: print plots in specified format(s) (space-separated) (default = ['png'])

This uses the saved `train_uniform_reports.pkl` output file.
