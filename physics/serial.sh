#!/bin/bash
#$ -N NORMAL
#$ -q free64

#$ -t 1-400
python toy_fc.py $SGE_TASK_ID normal /data/users/linggel/normal_contour/
