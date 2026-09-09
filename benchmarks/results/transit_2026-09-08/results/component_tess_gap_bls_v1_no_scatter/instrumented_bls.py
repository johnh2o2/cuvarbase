def _eebls_gpu_fast_impl(t, y, dy, freqs, fname, use_optimized, qmin=0.01, qmax=0.5, ignore_negative_delta_sols=False, functions=None, stream=None, dlogq=0.3, memory=None, noverlap=2, max_nblocks=5000, force_nblocks=None, dphi=0.0, shmem_lim=None, freq_batch_size=None, transfer_to_device=True, transfer_to_host=True, convention='chi2ratio', **kwargs):
    """Shared implementation behind :func:`eebls_gpu_fast` and
    :func:`eebls_gpu_fast_optimized`; see their docstrings for the
    parameter descriptions."""
    _name = 'eebls_gpu_fast_optimized' if use_optimized else 'eebls_gpu_fast'
    check_lightcurve(t, y, dy, min_n=_BLS_MIN_NDATA, name=_name)
    check_freqs(freqs, name=_name)
    _validate_fast_q_bounds(len(freqs), qmin, qmax)
    _validate_noverlap(noverlap)
    _validate_convention(convention)
    if convention != 'chi2ratio' and (not transfer_to_host):
        raise ValueError("convention=%r requires transfer_to_host=True (the device-side periodogram is always 'chi2ratio')" % (convention,))
    if functions is None:
        if kwargs.get('prepare', True):
            functions = _get_cached_kernels(kwargs.get('block_size', _default_block_size), use_optimized, [fname, 'full_bls_no_sol_fused'])
        else:
            ckw = dict(kwargs)
            ckw.setdefault('use_optimized', use_optimized)
            functions = compile_bls(function_names=[fname, 'full_bls_no_sol_fused'], **ckw)
    func = functions[fname]
    fused_func = None
    try:
        fused_func = functions.get('full_bls_no_sol_fused')
    except AttributeError:
        fused_func = None
    noverlap_int = int(noverlap)
    use_fused = fused_func is not None and noverlap_int >= 2 and (float(dphi) == 0.0) and (noverlap_int & noverlap_int - 1 == 0)
    if shmem_lim is None:
        att = cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        shmem_lim = ensure_context().device.get_attribute(att)
    if memory is None:
        memory = None
        if stream is None and transfer_to_host and ('max_ndata' not in kwargs) and ('max_nfreqs' not in kwargs):
            memory = _pooled_bls_memory(t, y, dy, qmin, qmax, freqs, kwargs)
        if memory is None:
            memory = BLSMemory.fromdata(t, y, dy, qmin=qmin, qmax=qmax, freqs=freqs, stream=stream, transfer=True, **kwargs)
    elif transfer_to_device:
        memory.setdata(t, y, dy, qmin=qmin, qmax=qmax, freqs=freqs, transfer=True, **kwargs)
    float_size = np.float32(1).nbytes
    block_size = kwargs.get('block_size', _default_block_size)
    auto_freq_batch = freq_batch_size is None
    if freq_batch_size is None:
        freq_batch_size = len(freqs)
    block = (block_size, 1, 1)
    qmin_min = 2 * float_size / (shmem_lim - float_size * block_size)
    if use_fused:
        with _BLS_PROFILE.part('Host maximum-bin scan'):
            global_max_nbins = int(np.max(memory.nbinsf[:len(freqs)]))
        fused_req = (block_size + 2 * noverlap_int * global_max_nbins) * float_size
        if fused_req > shmem_lim:
            use_fused = False
        elif auto_freq_batch and _shmem_limits_occupancy(fused_req, block_size):
            freq_batch_size = _OCCUPANCY_FREQ_CHUNK
    best_bls_g = None
    n_passes = 1 if use_fused else noverlap
    for i_pass in range(n_passes):
        dphi_pass = dphi + float(i_pass) / noverlap
        i_freq = 0
        while i_freq < len(freqs):
            j_freq = min([i_freq + freq_batch_size, len(freqs)])
            nfreqs = j_freq - i_freq
            with _BLS_PROFILE.part('Host maximum-bin scan'):
                max_nbins = int(np.max(memory.nbinsf[i_freq:j_freq]))
            if use_fused:
                hist_size = noverlap_int * int(max_nbins)
            else:
                hist_size = int(max_nbins)
            mem_req = (block_size + 2 * hist_size) * float_size
            if mem_req > shmem_lim:
                s = 'qmin = %.2e requires too much shared memory.' % (1.0 / max_nbins)
                s += ' Either try a larger value of qmin (> %e)' % qmin_min
                s += ' or avoid using %s.' % ('eebls_gpu_fast_optimized' if use_optimized else 'eebls_gpu_fast')
                raise ValueError(s)
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
            if use_fused:
                args += (np.uint32(hist_size), np.uint32(noverlap_int))
                args += (np.float32(dlogq), np.float32(dphi))
            else:
                args += (np.uint32(max_nbins), np.uint32(1))
                args += (np.float32(dlogq), np.float32(dphi_pass))
            args += (np.uint32(ignore_negative_delta_sols),)
            launch_func = fused_func if use_fused else func
            if stream is not None:
                launch_func.prepared_async_call(*args, shared_size=int(mem_req))
            else:
                launch_func.prepared_call(*args, shared_size=int(mem_req))
            i_freq = j_freq
        if not use_fused and noverlap > 1:
            if best_bls_g is None:
                best_bls_g = memory.bls_g.copy()
            else:
                gpuarray.maximum(memory.bls_g, best_bls_g, out=best_bls_g, stream=stream)
    if best_bls_g is not None:
        cuda.memcpy_dtod(memory.bls_g.gpudata, best_bls_g.gpudata, best_bls_g.nbytes)
    if transfer_to_host:
        memory.transfer_data_to_cpu()
        if stream is not None:
            stream.synchronize()
        chi2_0 = getattr(memory, 'chi2_0', None)
        if chi2_0 is None:
            chi2_0 = _chi2_null(y, dy)
        return _convert_bls_power_from_chi2_0(memory.bls, chi2_0, convention)
    return memory.bls
