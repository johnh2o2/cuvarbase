import subprocess, os
from cuvarbase.utils import find_kernel, _module_reader
for dbl in (False, True):
    d = dict(NPHASE=5, NMAG=4, PHASE_OVERLAP=0, MAG_OVERLAP=0)
    if dbl: d['DOUBLE_PRECISION'] = None
    txt = _module_reader(find_kernel('ce'), cpp_defs=d)
    open('/workspace/scratch/ce_ptx.cu', 'w').write(txt)
    subprocess.check_call(['nvcc', '-ptx', '-arch=sm_89', '--use_fast_math', '-o', '/workspace/scratch/ce_ptx.ptx', '/workspace/scratch/ce_ptx.cu'])
    for line in open('/workspace/scratch/ce_ptx.ptx'):
        if '.shared' in line and ('sh[' in line or 'f0' in line):
            print('double=%s: %s' % (dbl, line.strip()))
