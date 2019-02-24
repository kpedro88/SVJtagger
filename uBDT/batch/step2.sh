#!/bin/bash

export JOBNAME=""
export PROCESS=""
export INPUT=""
export OUTDIR=""
export OPTIND=1
while [[ $OPTIND -lt $# ]]; do
	# getopts in silent mode, don't exit on errors
	getopts ":j:p:i:o:" opt || status=$?
	case "$opt" in
		j) export JOBNAME=$OPTARG
		;;
		p) export PROCESS=$OPTARG
		;;
		o) export OUTDIR=$OPTARG
		;;
		i) export INPUT=$OPTARG
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
echo ""

# pick out config for this job
CONFIGS=()
IFS="," read -a CONFIGS <<< "$INPUT"
CONFIG=${CONFIGS[$PROCESS]}

cd $CMSSW_BASE/src/SVJtagger/uBDT

# run training
THREADS=$(getFromClassAd RequestCpus)
echo "python train_uniform.py -C $CONFIG -d trainings/$CONFIG -t $THREADS -v"
python train_uniform.py -C $CONFIG -d trainings/$CONFIG -t $THREADS -v

TRAINEXIT=$?

if [[ $TRAINEXIT -ne 0 ]]; then
	echo "exit code $TRAINEXT, skipping xrdcp
	exit $TRAINEXIT
fi

# tar output
OUTFILE=trainings_${CONFIG}.tar.gz
tar -czf $OUTFILE trainings/$CONFIG -C trainings

XRDEXIT=$?
if [[ $XRDEXIT -ne 0 ]]; then
	rm ${OUTFILE}
	echo "exit code $XRDEXIT, failure in xrdcp"
	exit $XRDEXIT
fi

