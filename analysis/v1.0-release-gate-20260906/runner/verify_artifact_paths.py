import subprocess
for name in ['wheel-smoke', 'sdist-smoke', 'wheel-test', 'sdist-test']:
    prefix='/tmp/cuvarbase-phase5/venvs/'+name+'/'
    code='import cuvarbase; print(cuvarbase.__file__); assert cuvarbase.__file__.startswith('+repr(prefix)+')'
    subprocess.run([prefix+'bin/python', '-c', code],check=True)
