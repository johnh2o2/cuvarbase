#!/bin/sh
cd /workspace/scratch
python oob_run.py default > oob_default.log 2>&1
python oob_run.py plain 0.1 0.5 5.4e9 plain_a.npy > oob_plain_a.log 2>&1
python oob_run.py padded 0.1 0.5 5.4e9 padded_a.npy > oob_padded_a.log 2>&1
python oob_run.py plain 0.05 0.5 1.4e10 plain_b.npy > oob_plain_b.log 2>&1
python oob_run.py padded 0.05 0.5 1.4e10 padded_b.npy > oob_padded_b.log 2>&1
echo ALLDONE > oob_done.flag
