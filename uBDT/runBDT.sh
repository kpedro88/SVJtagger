#!/bin/bash

MODE=""
CONFIGS=()
CONFIGLIST=""
VERBOSE=""
GRIDVERSION=-1
QUERY=""
FETCH=0
PFORMAT=""

while getopts "m:C:G:q:vfp" opt; do
	case "$opt" in
		m) MODE=$OPTARG
		;;
		C) CONFIGLIST="$OPTARG"; IFS="," read -a CONFIGS <<< "$CONFIGLIST"
		;;
		G) GRIDVERSION=$OPTARG
		;;
		q) QUERY="$OPTARG"
		;;
		v) VERBOSE="-v"
		;;
		f) FETCH=1
		;;
		p) PFORMAT="-f png pdf"
		;;
	esac
done

TARDIR=/uscmst1b_scratch/lpc1/3DayLifetime/pedrok/bdt
EOSDIR=/store/user/pedrok/SVJ2017/ubdt/output
REDIR=root://cmseos.fnal.gov
if [ ! -d $TARDIR ]; then
	mkdir -p $TARDIR
fi

if [ "$MODE" = plot ]; then
	for CONFIG in ${CONFIGS[@]}; do
		echo "Config: $CONFIG"
		TRAINDIR=trainings_${CONFIG}
		# get files from batch job
		if [ ! -d ${TARDIR}/${TRAINDIR} ] || [ -z "$(ls -A ${TARDIR}/${TRAINDIR})" ]; then
			echo "fetching from EOS"
			cd ${TARDIR}
			xrdcp -s ${REDIR}/${EOSDIR}/${TRAINDIR}.tar.gz .
			tar -xzf ${TRAINDIR}.tar.gz
			rm ${TRAINDIR}.tar.gz
			cd -
		fi
		if [ "$FETCH" -eq 1 ]; then
			continue
		fi
		# check for existing plots
		PLOTDIR=plots_${CONFIG}
		OUTDIR=plots/${CONFIG}
		if [ -d ${TARDIR}/${PLOTDIR} ]; then
			cp -rT ${TARDIR}/${PLOTDIR} ${OUTDIR}
			echo "copied existing plots"
		else
			# bdt plots
			python report_uniform.py -C $CONFIG -d ${TARDIR}/${TRAINDIR} -o ${OUTDIR} -c bdt -t flat -s bdt $PFORMAT
			# ubdt plots
			python report_uniform.py -C $CONFIG -d ${TARDIR}/${TRAINDIR} -o ${OUTDIR} -c ubdt -t proc -s ubdt $PFORMAT
			echo ""
		fi
	done
elif [ "$MODE" = analyze ]; then
	GRIDARG=""
	PLOTDIR=plots
	if [ "$GRIDVERSION" -gt 0 ]; then
		GRIDARG="-G $GRIDVERSION"
		PLOTDIR=$TARDIR/grids/plots
		mkdir -p $PLOTDIR
		GRIDNAME=$(python -c 'from mods import config_path; config_path(); from makeGrid import getGridName; print getGridName("'$CONFIGLIST'",'$GRIDVERSION')')
		for i in $(eos $REDIR ls $EOSDIR/grids | grep $GRIDNAME); do
			POINTNAME=$(echo $i | sed 's/.tar.gz//; s/trainings_//')
			if [ ! -d ${PLOTDIR}/${POINTNAME} ] || [ -z "$(ls -A ${PLOTDIR}/${POINTNAME})" ]; then
				cd ${TARDIR}/grids
				xrdcp -s ${REDIR}/${EOSDIR}/grids/trainings_${POINTNAME}.tar.gz .
				tar -xzf trainings_${POINTNAME}.tar.gz
				mv plots_${POINTNAME} plots/${POINTNAME}
				rm trainings_${POINTNAME}.tar.gz
				cd -
			fi
		done
	fi
	# bdt analysis
	echo "bdt"
	python analyze_uniform.py -C $CONFIGLIST $GRIDARG -d $PLOTDIR -c bdt -s bdt -q "$QUERY"
	# ubdt analysis
	echo "ubdt"
	python analyze_uniform.py -C $CONFIGLIST $GRIDARG -d $PLOTDIR -c ubdt -s ubdt -q "$QUERY"
elif [ "$MODE" = upload ]; then
	COPIES=""
	for CONFIG in ${CONFIGS[@]}; do
		COPIES="$COPIES plots/${CONFIG}"
	done
	scp -r $COPIES pedrok@pedrok.phpwebhosting.com:/home/pedrok/www/analysis/svj/bdt/
fi
