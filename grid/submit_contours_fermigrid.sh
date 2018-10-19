#!/bin/bash


BASEDIR=$1
FCBASEDIR=`dirname "$0"`"/.."

slice_size=1600
# slice_types=("dcp__theta23_IH" "dcp__theta23_NH" "theta23__dmsq_32_NH" "theta23__dmsq_32_IH")
slice_types=("dcp__theta23_NH")
vars=("dcp__theta23" "theta23__dmsq_32")

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
  jobsub_cmd=$jobsub_cmd" --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC"
  jobsub_cmd=$jobsub_cmd" --disk=2000MB"
  jobsub_cmd=$jobsub_cmd" --memory=2000MB"
  jobsub_cmd=$jobsub_cmd" --expected-lifetime=86400s"
  jobsub_cmd=$jobsub_cmd" -f dropbox://"$FCBASEDIR"/physics/"$FCFILE" -f dropbox://"$FCBASEDIR"/physics/fc_helper.py -f dropbox://"$FCBASEDIR"/physics/toy_experiment.py"
  jobsub_cmd=$jobsub_cmd" file://"$FCBASEDIR"/grid/toyfc_grid_script.sh"
  jobsub_cmd=$jobsub_cmd" --fcfile "$FCFILE
  jobsub_cmd=$jobsub_cmd" --fctype "$fc_type
  jobsub_cmd=$jobsub_cmd" --ctype "$slice_var 
  jobsub_cmd=$jobsub_cmd" --outdir "$OUTDIR
  jobsub_cmd=$jobsub_cmd" --njobs "$slice_size
  # jobsub_cmd=$jobsub_cmd" --append_condor_requirements='(((TARGET.GLIDEIN_ToDie-CurrentTime)>86400s)||isUndefined(TARGET.GLIDEIN_ToDie))'"
  
  eval $jobsub_cmd
  echo "-------------"
done
