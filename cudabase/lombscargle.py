
class LSSpectrogram(BaseSpectrogram):
    def __init__(self, t, y, w, **kwargs):

        pdm_kwargs = dict( dphi = dphi, block_size = block_size, 
                                kind=kind, nbins=nbins )

        super(PDMSpectrogram, self).__init__(t, y, w, **kwargs)

        self.proc_kwargs.update(pdm_kwargs)
        if self.proc is None:
            self.proc = PDMAsyncProcess()


    def model(self, freq, time=None):
        t, y, w = self.t, self.y, self.w
        if not time is None:
            t, y, w = self.weighted_local_data(time)

        return binned_pdm_model(t, y, w, freq, self.proc_kwargs['nbins'], 
                            linterp=(self.proc_kwargs['kind'] =='binned_linterp'))




