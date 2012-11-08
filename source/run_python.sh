#!/bin/sh
#
# the next line is a "magic" comment that tells codine to use bash
#$ -S /bin/bash
#
# This script should be what is passed to qsub; its job is just to run one matlab job.

# export PYTHONPATH=/home/mlg/dkd23/local/lib/python2.5/site-packages:$PYTHONPATH

python /home/mlg/dkd23/git/gp-structure-search/source/structure_search.py $1 $2 $3 $4

