#!/usr/bin/env python3
"""BLS wall-phase profiles and public-API fusion ablation; no installed source edits."""
import argparse,ast,hashlib,inspect,json,sys,textwrap,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from worker import Backend,sha,dump
import worker


class Profiler:
    def __init__(self,sync):self.sync=sync;self.rows={};self.stack=[]
    @contextmanager
    def part(self,name):
        self.sync();state=[time.perf_counter(),0.];self.stack.append(state)
        try:yield
        finally:
            self.sync();elapsed=time.perf_counter()-state[0];self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            row=self.rows.setdefault(name,dict(calls=0,inclusive_s=0.,exclusive_s=0.))
            row['calls']+=1;row['inclusive_s']+=elapsed;row['exclusive_s']+=elapsed-state[1]
    def wrap(self,name,fn):
        def wrapped(*a,**kw):
            with self.part(name):return fn(*a,**kw)
        return wrapped


class Kernel:
    def __init__(self,fn,profile,name):self.fn=fn;self.profile=profile;self.name=name
    def __getattr__(self,name):
        value=getattr(self.fn,name)
        return self.profile.wrap('GPU kernel launches (synchronized)',value) if name in ['prepared_call','prepared_async_call'] else value


class Scans(ast.NodeTransformer):
    def visit_Assign(self,node):
        if ast.unparse(node.targets[0]) in ['max_nbins','global_max_nbins']:
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Attribute(value=ast.Name(id='_BLS_PROFILE',ctx=ast.Load()),attr='part',ctx=ast.Load()),
                args=[ast.Constant('Host maximum-bin scan')],keywords=[]))],body=[node]),node)
        return node


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=np.load(a.input);cfg=json.loads(a.config);meta=json.loads(str(d['metadata']));i=2
    lc=tuple(np.array(d[f'{k}_{i}']) for k in ['t','y','dy']);b=Backend(cfg,d,len(lc[0]));b.search([lc])
    times=[]
    for rep in range(3):
        b.sync();start=time.perf_counter();native=b.search([lc])[0];b.sync();times.append(time.perf_counter()-start)
    m=b.bls;p=Profiler(b.sync)
    # Rebuild exactly one function with two max-bin scans wrapped in timers.
    target='_eebls_gpu_fast_impl' if cfg['backend'].startswith('v1') else 'eebls_gpu_fast'
    old=getattr(m,target);source=inspect.getsource(old);tree=Scans().visit(ast.parse(textwrap.dedent(source)));ast.fix_missing_locations(tree)
    transformed=ast.unparse(tree)+'\n';path=a.out/'instrumented_bls.py';path.write_text(transformed)
    # Keep the original globals live, so wrapped lookup/memory methods below are resolved normally.
    m.__dict__['_BLS_PROFILE']=p;exec(compile(tree,str(path),'exec'),m.__dict__)
    for name,label in [('setdata','Host preparation and H2D'),('transfer_data_to_cpu','Spectrum D2H')]:
        setattr(m.BLSMemory,name,p.wrap(label,getattr(m.BLSMemory,name)))
    if hasattr(m,'_pooled_bls_memory'):
        m._pooled_bls_memory=p.wrap('Memory pool / allocation',m._pooled_bls_memory)
    if hasattr(m,'_get_cached_kernels'):
        original=m._get_cached_kernels
        def cached(*args,**kwargs):
            with p.part('Kernel cache lookup'):
                return {k:Kernel(fn,p,k) for k,fn in original(*args,**kwargs).items()}
        m._get_cached_kernels=cached
    if b.functions is not None:b.functions={k:Kernel(fn,p,k) for k,fn in b.functions.items()}
    worker.spectral_candidate=p.wrap('Common candidate ranking',worker.spectral_candidate)
    profiles=[]
    for rep in range(2):
        p.rows={}
        with p.part('Remaining public API work'):
            profiled=b.search([lc])[0]
        profiles.append(p.rows)
    gridtimes=[]
    for rep in range(3):
        start=time.perf_counter()
        f,q=m.transit_autofreq(lc[0],rho=1.,samples_per_peak=2,qmin_fac=.5,
            fmin=1/meta['pmax'],fmax=1/meta['pmin'])
        gridtimes.append(time.perf_counter()-start)
    # Actual source baseline can differ slightly after independently missing endpoint samples.
    fullbaseline=float(np.ptp(lc[0]))
    np.savez_compressed(a.out/'outputs.npz',native=native['power'],profiled=profiled['power'],periods=native['periods'],auto_freqs=f,auto_q=q)
    dump(a.out/'summary.json',dict(status='ok',profile=meta['profile'],config=cfg,input_sha256=sha(a.input),
        worker_sha256=sha(__file__),instrumented_sha256=sha(path),original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        native_times_s=times,native_median_s=float(np.median(times)),profiles=profiles,
        native_candidate=native['candidate'],profiled_candidate=profiled['candidate'],
        max_abs_power_difference=float(np.max(np.abs(native['power']-profiled['power']))),
        grid_times_s=gridtimes,grid_median_s=float(np.median(gridtimes)),grid_baseline=fullbaseline,n_auto_freqs=len(f),
        output_sha256=sha(a.out/'outputs.npz'),meaning='Synchronized wall regions, not kernel-busy traces. Grid construction is measured separately and excluded from explicit-grid API timings.'))
    print('BLS_COMPONENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
