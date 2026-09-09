def eebls_gpu_fast(t, y, dy, freqs, qmin=0.01, qmax=0.5, ignore_negative_delta_sols=False, functions=None, stream=None, dlogq=0.3, memory=None, noverlap=2, max_nblocks=5000, force_nblocks=None, dphi=0.0, shmem_lim=None, freq_batch_size=None, transfer_to_device=True, transfer_to_host=True, **kwargs):
    """
    Box-Least Squares with PyCUDA but about 2-3 orders of magnitude
    faster than eebls_gpu. Uses shared memory for the binned data,
    which means that there is a lower limit on the q values that
    this function can handle.

    To save memory and improve speed, the best solution is not
    kept. To get the best solution, run ``eebls_gpu`` at the
    optimal frequency.

    .. warning::

        If you are running on a single-GPU machine, there may be a
        kernel time limit set by your OS. If running this function
        produces a timeout error, try setting ``freq_batch_size`` to a
        reasonable number (~10). That will split up the computations by
        frequency.

    .. note::

        No extra global memory is needed, meaning you likely do *not* need
        to use ``large_run`` with this function.

    .. note::

        There is no ``noverlap`` parameter here yet. This is only a problem
        if the optimal ``q`` value is close to ``qmin``. To alleviate this,
        you can run this function ``noverlap`` times with
        ``dphi = i/noverlap`` for the ``i``-th run. Then take the best solution
        of all runs.

    Parameters
    ----------
    t: array_like, float
        Observation times
    y: array_like, float
        Observations
    dy: array_like, float
        Observation uncertainties
    freqs: array_like, float
        Frequencies
    qmin: float or array_like, optional (default: 1e-2)
        minimum q values to search at each frequency
    qmax: float or array_like (default: 0.5)
        maximum q values to search at each frequency
    ignore_negative_delta_sols: bool
        Whether or not to ignore solutions with a negative delta (i.e. an inverted dip)
    dphi: float, optional (default: 0.)
        Phase offset (in units of the finest grid spacing). If you
        want ``noverlap`` bins at the smallest ``q`` value, run this
        function ``noverlap`` times, with ``dphi = i / noverlap``
        for the ``i``-th run and take the best solution for all the runs.
    dlogq: float
        The logarithmic spacing of the q values to use. If negative,
        the q values increase by ``dq = qmin``.
    functions: dict
        Dictionary of compiled functions (see :func:`compile_bls`)
    freq_batch_size: int, optional (default: None)
        Number of frequencies to compute in a single batch; if
        ``None`` this will run a single batch for all frequencies
        simultaneously
    shmem_lim: int, optional (default: None)
        Maximum amount of shared memory to use per block in bytes.
        This is GPU-dependent but usually around 48KB. If ``None``,
        uses device information provided by PyCUDA (recommended).
    max_nblocks: int, optional (default: 200)
        Maximum grid size to use
    force_nblocks: int, optional (default: None)
        If this is set the gridsize is forced to be this value
    memory: :class:`BLSMemory` instance, optional (default: None)
        See :class:`BLSMemory`.
    transfer_to_host: bool, optional (default: True)
        Transfer BLS back to CPU.
    transfer_to_device: bool, optional (default: True)
        Transfer data to GPU
    **kwargs:
        passed to `compile_bls`

    Returns
    -------
    bls: array_like, float
        BLS periodogram, normalized to
        :math:`1 - \\chi_2(\\omega) / \\chi_2(constant)`

    """
    fname = 'full_bls_no_sol'
    if functions is None:
        functions = compile_bls(function_names=[fname], **kwargs)
    func = functions[fname]
    if shmem_lim is None:
        dev = pycuda.autoprimaryctx.device
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = pycuda.autoprimaryctx.device.get_attribute(att)
    if memory is None:
        memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax, freqs=freqs, stream=stream, transfer=True, **kwargs)
    elif transfer_to_device:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax, freqs=freqs, transfer=True, **kwargs)
    float_size = np.float32(1).nbytes
    block_size = kwargs.get('block_size', _default_block_size)
    if freq_batch_size is None:
        freq_batch_size = len(freqs)
    nbatches = int(np.ceil(len(freqs) / freq_batch_size))
    block = (block_size, 1, 1)
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)
    i_freq = 0
    while i_freq < len(freqs):
        j_freq = min([i_freq + freq_batch_size, len(freqs)])
        nfreqs = j_freq - i_freq
        with _BLS_PROFILE.part('Host maximum-bin scan'):
            max_nbins = max(memory.nbinsf[i_freq:j_freq])
        mem_req = (block_size + 2 * max_nbins) * float_size
        if mem_req > shmem_lim:
            s = 'qmin = %.2e requires too much shared memory.' % (1.0 / max_nbins)
            s += ' Either try a larger value of qmin (> %e)' % qmin_min
            s += ' or avoid using eebls_gpu_fast.'
            raise Exception(s)
        nblocks = min([nfreqs, max_nblocks])
        if force_nblocks is not None:
            nblocks = force_nblocks
        grid = (nblocks, 1)
        args = (grid, block)
        if stream is not None:
            args += (stream,)
        args += (memory.t_g.ptr, memory.yw_g.ptr, memory.w_g.ptr)
        args += (memory.bls_g.ptr, memory.freqs_g.ptr)
        args += (memory.nbins0_g.ptr, memory.nbinsf_g.ptr)
        args += (np.uint32(len(t)), np.uint32(nfreqs), np.uint32(i_freq))
        args += (np.uint32(max_nbins), np.uint32(noverlap))
        args += (np.float32(dlogq), np.float32(dphi))
        args += (np.uint32(ignore_negative_delta_sols),)
        if stream is not None:
            func.prepared_async_call(*args, shared_size=int(mem_req))
        else:
            func.prepared_call(*args, shared_size=int(mem_req))
        i_freq = j_freq
    if transfer_to_host:
        memory.transfer_data_to_cpu()
        if stream is not None:
            stream.synchronize()
    return memory.bls
