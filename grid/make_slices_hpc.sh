#!/bin/bash

if [ -d ./temp ]; then rm -rfv ./temp; fi
slice_size=40
slice_types=("dcp_NHUO" "dcp_NHLO" "dcp_NH" "dcp_IHUO" "dcp_IHLO" "dcp_IH" "dmsq_32_NHUO" "dmsq_32_NHLO" "dmsq_32_NH" "dmsq_32_IHUO" "dmsq_32_IHLO" "dmsq_32_IH" "theta23_NH" "theta23_IH")
vars=("dcp" "theta23" "dmsq_32")
mkdir temp
for stype in "${slice_types[@]}"; do
  fc_type=""
  slice_var=""
  for var in "${vars[@]}"; do
    if [ "${stype/$var}" != $stype ]; then
      fc_type="${stype/$var"_"}"
      slice_var=$var
    fi
  done
  if [ ! -d /data/users/nayakb/fc_results/$stype/ ]; then 
    mkdir -p /data/users/nayakb/fc_results/$stype/
  fi
  cat <<EOF >> temp/${stype}.sh
  #!/bin/bash
  #$ -N $stype
  #$ -q free64

  #$ -t 1-$slice_size
  python fc.py \$SGE_TASK_ID "$fc_type" "$slice_var" /data/users/nayakb/fc_results/$stype/
EOF
echo "Submitting $slice_size jobs for $stype to /data/users/nayakb/fc_results/$stype/ ..."
qsub ./temp/${stype}.sh
done
