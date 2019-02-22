# SVJ tagger: uBDT

This is a uniform gradient-boosted BDT using flattening loss from hep_ml.

## Setup

```
wget https://raw.githubusercontent.com/kpedro88/SVJtagger/master/uBDT/setup.sh
chmod +x setup.sh
./setup.sh
cd CMSSW_10_3_0/src/SVJtagger/uBDT
cmsenv
```

## Running

```
python train_uniform.py
```

Arguments:
* `-C, --config [file]`: config to provide parameters (default: test1)
* `-t, --train-test-size [size]`: size for test and train datasets (override config) (default: -1)
* `-d, --dir [dir]`: directory for output files (required)
* `-v, --verbose`: enable message printing (default: False)

For rapid prototyping, use `-t 0.01`.

Output files produced:
* `[dir]/train_uniform_classifiers.pkl`: classifiers in python pickle format
* `[dir]/train_uniform_reports.pkl`: report in python pickle format
* `[dir]/TMVA_bdt_weights.xml`: BDT in TMVA format
* `[dir]/TMVA_ubdt_weights.xml`: uBDT in TMVA format

## Producing plots

```
python report_uniform.py
```

Arguments:
* `-d, --dir [dir]`: directory for train_uniform_reports.pkl file (required)
* `-o, --outdir [dir]`: directory for output pngs (if different from input dir)
* `-C, --config [file]`: config to provide parameters (default: test1)
* `-c, --classifiers [list]`: plot only for specified classifier(s) (space-separated) (default = [] -> all)
* `-t, --test {flat,proc}`: suffix for report names (test*, train*)
* `-s, --suffix [suffix]`: suffix for plots
* `-f, --formats [list]`: print plots in specified format(s) (space-separated) (default = ['png'])
* `-v, --verbose`: enable message printing (default = False)

This uses the saved `[dir]/train_uniform_reports.pkl` output file.
