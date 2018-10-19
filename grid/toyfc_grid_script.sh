#!/bin/bash
exec &> >(tee -a log.txt)

echo `whoami`@`hostname`:`pwd` at `date`

copy_log()
{
    infix=""
    if [ $NJOBS -gt 1 ]; then
    	infix="${SUFFIX}."
    fi

    target="$OUTDIR/${JOBSUBJOBID}.${infix}log.txt"
    echo Copy log to $target
    ifdh cp $PWD/log.txt $target
}

# Make sure we always attempt to copy the logs
abort()
{
    copy_log
    exit 3
}

while [[ $1 == *toyfc_grid_script.sh ]]
do
    echo Braindead scripting passed us our own script name as first argument, discarding
    shift 1
done

function setup_pyroot {
  source /cvmfs/nova.opensciencegrid.org/externals/setup
  setup python v2_7_14b -f Linux64bit+2.6-2.12 -z /cvmfs/nova.opensciencegrid.org/externals
  setup numpy v1_14_1 -f Linux64bit+2.6-2.12 -q e15:p2714b:prof -z /cvmfs/nova.opensciencegrid.org/externals
  setup scipy v0_15_0a -f Linux64bit+2.6-2.12 -q e7:prof -z /cvmfs/nova.opensciencegrid.org/externals
  setup root v6_12_06a -f Linux64bit+2.6-2.12 -z /cvmfs/nova.opensciencegrid.org/externals -qe15:prof
  setup jobsub_client v1_2_8_2  -f NULL -z /cvmfs/fermilab.opensciencegrid.org/products/common/db
  setup ifdhc v2_3_3 -f Linux64bit+2.6-2.12  -q e15:p2714b:prof -z /cvmfs/nova.opensciencegrid.org/externals
  setup ifdhc_config  v2_3_3 -f NULL
}

while [ $# -gt 0 ]
do
    echo "args: \$1 '$1' \$2 '$2'"
    case $1 in
        -m | --fcfile )
            FCFILE=$2
            shift 2
            ;;
        -t | --fctype )
            FCTYPE=$2
            shift 2
            ;;
        -c | --ctype )
            CONTOURTYPE=$2
            shift 2
            ;;
        -o | --outdir )
            OUTDIR=$2
            shift 2
            ;;
        -n | --njobs )
            NJOBS=$2
            shift 2
            ;;
    esac
done

let COUNT=PROCESS+1
SUFFIX=${COUNT}_of_${NJOBS}

echo "Setting up environment.."
setup_pyroot

PWD=`pwd`
echo "Current directory: $PWD"
TEMPOUT=$PWD"/output/"
mkdir -p $TEMPOUT
cd $TEMPOUT

echo "-----------------------------------------------"
echo "Environment at job start:"
printenv
echo
echo
echo "Active UPS products:"
ups active
echo "-----------------------------------------------"

echo list condor input dir, CONDOR_DIR_INPUT:
echo 'ls -a '${CONDOR_DIR_INPUT}
ls -a ${CONDOR_DIR_INPUT}

FCFILE=${CONDOR_DIR_INPUT}/$FCFILE
FCHELPER=${CONDOR_DIR_INPUT}/fc_helper.py
FITTER=${CONDOR_DIR_INPUT}/toy_experiment.py

if [ ! -f $FCFILE ]
then
    echo FCFILE $FCFILE somehow not copied to $CONDOR_DIR_INPUT
    abort
fi

if [ ! -f $FCHELPER ]
then
    echo FCHELPER $FCHELPER somehow not copied to $CONDOR_DIR_INPUT
    abort
fi

if [ ! -f $FITTER ]
then
    echo FITTER $FITTER somehow not copied to $CONDOR_DIR_INPUT
    abort
fi

exit_flag=0
echo "Running FC job.."
CMD="python $FCFILE $COUNT $FCTYPE $CONTOURTYPE $TEMPOUT"
echo $CMD
time $CMD || exit_flag=$?

ls -lh

echo "Files in output directory to be copied back to ${OUTDIR} are:"
ls
echo

if [[ `ls | wc -l` == 0 ]]
then
    echo Nothing to copy back
    abort
fi

for k in *
do
    if [ $NJOBS -gt 1 ]
    then
    # Insert the suffix after the last dot
	DEST=`echo $k | sed "s/\(.*\)\.\(.*\)/\1.${SUFFIX}.\2/"`
	if [ $DEST = $k ]
	then
          # $k had no dots in it?
          DEST=${k}.$SUFFIX
	fi
    else
	DEST=$k
    fi
    CMD="ifdh cp $k $OUTDIR/$DEST"
    echo $CMD
    $CMD || exit_flag=$?
done

echo Done!
copy_log
exit $exit_flag
echo also done copying log. Will now exit.
