import numpy as np, multiprocessing as mp
def w(_):
    from cuvarbase.bls import eebls_gpu_fast
    t=np.linspace(0,100,600); y=1+0.01*np.sin(t); dy=0.01*np.ones(600)
    return float(eebls_gpu_fast(t,y,dy,np.linspace(0.1,3,200)).max())
if __name__=="__main__":
    with mp.get_context("spawn").Pool(2) as p: print("spawn pool:", p.map(w,[0,1]))