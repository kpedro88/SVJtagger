#!/bin/bash

export JOBNAME=""
export PROCESS=""
export INPUT=""
export OUTDIR=""
export OPTIND=1
export DISCARD=0
export GRID=0
while [[ $OPTIND -lt $# ]]; do
	# getopts in silent mode, don't exit on errors
	getopts ":j:p:i:o:dg" opt || status=$?
	case "$opt" in
		j) export JOBNAME=$OPTARG
		;;
		p) export PROCESS=$OPTARG
		;;
		o) export OUTDIR=$OPTARG
		;;
		i) export INPUT=$OPTARG
		;;
		d) export DISCARD=1
		;;
		g) export GRID=1
		;;
		# keep going if getopts had an error
		\? | :) OPTIND=$((OPTIND+1))
		;;
	esac
done

echo "parameter set:"
echo "INPUT:      $INPUT"
echo "OUTDIR:     $OUTDIR"
echo "JOBNAME:    $JOBNAME"
echo "PROCESS:    $PROCESS"
echo "DISCARD:    $DISCARD"
echo ""

cd $CMSSW_BASE/src/SVJtagger/uBDT

# pick out config for this job
CONFIG=""
if [ "$GRID" -eq 0 ]; then
	CONFIGS=()
	IFS="," read -a CONFIGS <<< "$INPUT"
	CONFIG=${CONFIGS[$PROCESS]}
else
	CONFIG=$(python -c 'from mods import config_path; config_path(); from makeGrid import getPointName; print getPointName('$PROCESS',"'$INPUT'")')
fi

# run training
OUTFILES=trainings_${CONFIG}
THREADS=$(getFromClassAd RequestCpus)
echo "python train_uniform.py -C $CONFIG -d $OUTFILES -t $THREADS -v"
python train_uniform.py -C $CONFIG -d $OUTFILES -t $THREADS -v

TRAINEXIT=$?

if [[ $TRAINEXIT -ne 0 ]]; then
	echo "exit code $TRAINEXIT, skipping xrdcp"
	exit $TRAINEXIT
fi

PLOTFILES=""
# make the plots and discard the big report pkl files
if [ "$DISCARD" -eq 1 ]; then
	PLOTFILES=plots_${CONFIG}
	# bdt plots
	echo "python report_uniform.py -C $CONFIG -d $OUTFILES -o $PLOTFILES -c bdt -t flat -s bdt"
	python report_uniform.py -C $CONFIG -d $OUTFILES -o $PLOTFILES -c bdt -t flat -s bdt
	# ubdt plots
	echo "python report_uniform.py -C $CONFIG -d $OUTFILES -o $PLOTFILES -c ubdt -t proc -s ubdt"
	python report_uniform.py -C $CONFIG -d $OUTFILES -o $PLOTFILES -c ubdt -t proc -s ubdt
	# remove big files
	rm $OUTFILES/*.pkl
fi

# tar output
tar -czf ${OUTFILES}.tar.gz $OUTFILES $PLOTFILES
xrdcp -f ${OUTFILES}.tar.gz ${OUTDIR}/${OUTFILES}.tar.gz

XRDEXIT=$?
if [[ $XRDEXIT -ne 0 ]]; then
	rm ${OUTFILES}.tar.gz
	echo "exit code $XRDEXIT, failure in xrdcp"
	exit $XRDEXIT
fi

