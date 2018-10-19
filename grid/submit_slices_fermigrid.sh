#!/bin/bash


BASEDIR=$1
FCBASEDIR=`dirname "$0"`"/.."

slice_size=40
slice_types=("dcp_NHUO" "dcp_NHLO" "dcp_NH" "dcp_IHUO" "dcp_IHLO" "dcp_IH" "dmsq_32_NHUO" "dmsq_32_NHLO" "dmsq_32_NH" "dmsq_32_IHUO" "dmsq_32_IHLO" "dmsq_32_IH" "theta23_NH" "theta23_IH")
vars=("dcp" "theta23" "dmsq_32")

FCFILE='fc.py'

for stype in "${slice_types[@]}"; do
  fc_type=""
  slice_var=""
  for var in "${vars[@]}"; do
    if [ "${stype/$var}" != $stype ]; then
      fc_type="${stype/$var"_"}"
      slice_var=$var
    fi
  done
  OUTDIR=$BASEDIR"/"$stype"/"
  if [ ! -d $OUTDIR ]; then 
    mkdir -p $OUTDIR
    chmod oug=wrx $OUTDIR
  fi
  echo "-------------"
  echo "Submitting $slice_size jobs for $stype to $OUTDIR ..."
  echo "-------------"
  jobsub_cmd="jobsub_submit -G nova" 
  jobsub_cmd=$jobsub_cmd" -N "$slice_size 
  jobsub_cmd=$jobsub_cmd" -f "$FCBASEDIR"/physics/"$FCFILE" -f "$FCBASEDIR"/physics/fc_helper.py -f "$FCBASEDIR"/physics/toy_experiment.py" 
  jobsub_cmd=$jobsub_cmd" file://"$FCBASEDIR"/grid/toyfc_grid_script.sh"
  jobsub_cmd=$jobsub_cmd" --fcfile "$FCFILE
  jobsub_cmd=$jobsub_cmd" --fctype "$fc_type
  jobsub_cmd=$jobsub_cmd" --ctype "$slice_var 
  jobsub_cmd=$jobsub_cmd" --outdir "$OUTDIR
  jobsub_cmd=$jobsub_cmd" --njobs "$slice_size
  jobsub_cmd=$jobsub_cmd" --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC"
  jobsub_cmd=$jobsub_cmd" --disk=2000"
  jobsub_cmd=$jobsub_cmd" --memory=2000"
  jobsub_cmd=$jobsub_cmd" --expected-lifetime=86400s"
  # jobsub_cmd=$jobsub_cmd" --append_condor_requirements='(((TARGET.GLIDEIN_ToDie-CurrentTime)>86400s)||isUndefined(TARGET.GLIDEIN_ToDie))'"
  
  eval $jobsub_cmd
  echo "-------------"
done
