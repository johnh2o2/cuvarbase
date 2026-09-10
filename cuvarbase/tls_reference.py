"""Observation-level TLS search using the pinned GTLS numerical objective.

The standard frontend lives in cuvarbase.tls. This module keeps the complete
search domain independent of GPU workspace chunks.
"""
from collections import OrderedDict
import operator
import threading

import numpy as np
import cupy as cp

from . import tls_reference_math as reference
from .tls_reference_prefix import NativePrefixPlan
from .utils import find_kernel


_MODULES = None
_PREFIX_PLANS = threading.local()
_PREFIX_CACHE_BYTES = 64 * 1024**2
_WORKSPACE_BYTES = 512 * 1024**2


def _row_flux_prefix(flux):
    """Literal native scans, also used when a graph cannot fit the cache."""
    prefix = cp.empty_like(flux)
    for row in range(len(flux)):
        cp.cumsum(flux[row], out=prefix[row])
    return prefix


class _PrefixPlanCache:
    """One thread's bounded LRU of exact native scan graphs."""

    def __init__(self, max_plans=4, max_bytes=_PREFIX_CACHE_BYTES):
        self.max_plans = operator.index(max_plans)
        self.max_bytes = operator.index(max_bytes)
        if self.max_plans < 1 or self.max_bytes < 1:
            raise ValueError('Native prefix cache limits must be positive')
        self.plans = OrderedDict()

    @property
    def owned_bytes(self):
        return sum(plan.owned_bytes for plan in self.plans.values())

    def _evict(self):
        _, plan = self.plans.popitem(last=False)
        # raw_search downloads each microchunk before another plan can be
        # evicted. Synchronization also makes direct serial helper use safe.
        plan.close()

    def prefix(self, flux):
        key = NativePrefixPlan.key_for(flux.shape)
        plan = self.plans.pop(key, None)
        if plan is not None:
            self.plans[key] = plan
            return plan(flux)
        required = NativePrefixPlan.buffer_bytes_for(flux.shape)
        if required >= self.max_bytes:
            return _row_flux_prefix(flux)
        while self.plans and (len(self.plans) >= self.max_plans or
                              self.owned_bytes + required >= self.max_bytes):
            self._evict()
        try:
            plan = NativePrefixPlan(flux.shape,
                                    max_bytes=self.max_bytes - self.owned_bytes)
        except MemoryError:
            # The small CUB workspace is known only after warming the scan.
            # Release older shapes and retry with the whole bounded budget.
            if not self.plans:
                return _row_flux_prefix(flux)
            self.close()
            try:
                plan = NativePrefixPlan(flux.shape, max_bytes=self.max_bytes)
            except MemoryError:
                return _row_flux_prefix(flux)
        self.plans[key] = plan
        return plan(flux)

    def close(self):
        while self.plans:
            self._evict()


def _native_flux_prefix(flux):
    cache = getattr(_PREFIX_PLANS, 'cache', None)
    if cache is None:
        cache = _PREFIX_PLANS.cache = _PrefixPlanCache()
    return cache.prefix(flux)


def _physical_chunk_plan(ndata, stride, nwidths, ntiles, requested, *,
                         full=False, free_bytes=None):
    """Bound physical rows without changing logical width unions or order.

    The estimate includes sorting/preparation scratch, retained prefix buffers,
    tile outputs and, in full mode, the three-dimensional OOTR arrays. Reserving
    at most a quarter of free device memory leaves headroom for the allocator,
    input/cache storage and CUDA's internal workspaces.
    """
    requested = operator.index(requested)
    if requested < 1:
        raise ValueError('work_chunk must be a positive integer')
    if free_bytes is None:
        free_bytes = cp.cuda.runtime.memGetInfo()[0]
    budget = min(_WORKSPACE_BYTES, int(free_bytes) // 4)
    per_row = 96 * int(stride) + 24 * int(ntiles) + 256
    if full:
        per_row += 12 * int(nwidths) * int(ndata)
    rows = min(requested, budget // per_row)
    if rows < 1:
        raise MemoryError('One TLS period requires an estimated {} bytes, '
                          'exceeding the {}-byte physical workspace budget'
                          .format(per_row, budget))
    return dict(rows=int(rows), budget_bytes=budget,
                estimated_bytes_per_row=per_row,
                estimated_chunk_bytes=int(rows) * per_row)


def modules():
    global _MODULES
    if _MODULES is None:
        with open(find_kernel('tls_reference_prepare')) as source:
            prep = cp.RawModule(code=source.read())
        with open(find_kernel('tls_reference')) as source:
            search = cp.RawModule(code=source.read())
        prep.compile()
        search.compile()
        _MODULES = prep, search
    return _MODULES


def native_group_size(nperiods, ndata, cache, free_bytes=None):
    """Reference's physical allocation rule, for exact run reconstruction."""
    if free_bytes is None:
        free_bytes = cp.cuda.runtime.memGetInfo()[0]
    stride = ndata + cache['padded_data_width']
    d = len(cache['widths'])
    limit = free_bytes / (5 * (stride * 2 + 2 + d * stride * 4 + 2 * d))
    size = int(min(np.floor(limit), nperiods / 30))
    if size < 15:
        size = int(size / 1.1)
    return max(1, size)


def _tiles(widths, ndata, skip_factor):
    durations, starts = [], []
    for d, width in enumerate(widths):
        skip = max(int(width) // skip_factor, 1)
        count = (ndata + skip - 1) // skip
        first = np.arange(0, count, 256, dtype=np.int32)
        durations.extend([d] * len(first))
        starts.extend(first)
    return cp.asarray(durations, dtype=cp.int32), cp.asarray(starts, dtype=cp.int32)


def raw_search(periods, t, y, dy, cache, *, group_size=None,
               work_chunk=256, skip_factor=8, transit_depth_min=1e-5,
               native_prefix=True, capture=False, full=False,
               duration_selection=None):
    """Search prepared native inputs; retain group unions across work chunks.

    All inputs use reference preprocessing, including rescaled errors. This
    function does not normalize scores or select cross-period candidates.
    Its grouping is explicit: changing work_chunk cannot change trial widths.
    work_chunk is an upper bound; actual physical chunks are limited by the
    estimated workspace and recorded in the result.
    Explicit duration_selection uses consecutive runs of identical bounds so
    no other period can broaden a caller's requested duration interval.
    """
    prep, scan = modules()
    periods = np.ascontiguousarray(periods, dtype=np.float64)
    t, y, dy = (np.asarray(v) for v in (t, y, dy))
    ndata, nperiods = len(t), len(periods)
    if group_size is None:
        group_size = max(1, int(nperiods / 30))
        if group_size < 15:
            group_size = max(1, int(group_size / 1.1))
    widths = np.asarray(cache['widths'], dtype=np.int32)
    pad = int(cache['padded_data_width'])
    stride = ndata + pad
    template_stride = cache['template_deficits'].shape[1]
    t_gpu = cp.asarray(t, dtype=cp.float64)
    y_gpu, dy_gpu = cp.asarray(y, dtype=cp.float32), cp.asarray(dy, dtype=cp.float32)
    p_gpu = cp.asarray(periods)
    n_gpu = cp.asarray([ndata], dtype=cp.int32)
    span_gpu = cp.asarray([np.ptp(t)], dtype=cp.float32)
    np_gpu = cp.asarray([nperiods], dtype=cp.int32)
    pad_gpu = cp.asarray([pad], dtype=cp.int32)
    stride_gpu = cp.asarray([stride], dtype=cp.int32)
    if duration_selection is None:
        minimum_gpu, maximum_gpu = cp.empty(nperiods, cp.int32), cp.empty(nperiods, cp.int32)
        prep.get_function('durationsGrid')(((nperiods + 255) // 256,), (256,),
            (p_gpu, maximum_gpu, minimum_gpu, span_gpu, n_gpu, np_gpu))
        minima, maxima = minimum_gpu.get(), maximum_gpu.get()
        width_masks = reference.chunk_width_masks(widths, minima, maxima, group_size)
        group_ranges = [(first, min(first + group_size, nperiods))
                        for first in range(0, nperiods, group_size)]
    else:
        minima = np.asarray(duration_selection['width_minima'], dtype=np.int32)
        maxima = np.asarray(duration_selection['width_maxima'], dtype=np.int32)
        if minima.shape != periods.shape or maxima.shape != periods.shape:
            raise ValueError('Explicit duration bounds must align with periods')
        if np.any(minima > maxima):
            raise ValueError('Explicit duration minima must not exceed maxima')
        changed = np.flatnonzero((minima[1:] != minima[:-1]) |
                                 (maxima[1:] != maxima[:-1])) + 1
        boundaries = np.r_[0, changed, nperiods]
        group_ranges = np.column_stack((boundaries[:-1], boundaries[1:]))
        # An explicit interval can differ at every period. Keep only its two
        # bounds and construct one width mask at a time, rather than retaining
        # a potentially enormous number-of-periods by number-of-widths table.
        width_masks = None
    result = dict(chi2=np.full(nperiods, np.nan, dtype=np.float32),
                  start=np.full(nperiods, -1, dtype=np.int32),
                  start_time=np.full(nperiods, np.nan, dtype=np.float64),
                  width_index=np.full(nperiods, -1, dtype=np.int32),
                  width=np.zeros(nperiods, dtype=np.int32),
                  depth=np.zeros(nperiods, dtype=np.float32),
                  group_size=int(group_size), work_chunk=int(work_chunk),
                  minima=minima, maxima=maxima, width_masks=width_masks,
                  group_ranges=np.asarray(group_ranges, dtype=np.int64),
                  skip_factor=int(skip_factor), stages={},
                  work_chunk_plans=[], work_chunks=[])
    captured = []
    for group, (first, last) in enumerate(group_ranges):
        mask = ((widths >= minima[first]) & (widths <= maxima[first])
                if duration_selection is not None else width_masks[group])
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        use_widths = widths[ids]
        nw = len(ids)
        w_gpu = cp.asarray(use_widths)
        templates_gpu = cp.asarray(cache['template_deficits'][ids])
        overshoot_gpu = cp.asarray(cache['overshoot'][ids])
        tile_duration, tile_first = _tiles(use_widths, ndata, 2**30 if full else skip_factor)
        nt = len(tile_duration)
        plan = _physical_chunk_plan(ndata, stride, nw, nt, work_chunk, full=full)
        result['work_chunk_plans'].append(dict(group=group, first=first,
                                               last=last, **plan))
        for start in range(first, last, plan['rows']):
            stop = min(start + plan['rows'], last)
            rows = stop - start
            result['work_chunks'].append(dict(group=group, start=start,
                                              stop=stop, rows=rows))
            rows_gpu = cp.asarray([rows], dtype=cp.int32)
            phases = cp.empty((rows, ndata), dtype=cp.float64)
            prep.get_function('foldFast')(((ndata + 255) // 256, rows), (256,),
                (t_gpu, p_gpu[start:stop], phases, rows_gpu, n_gpu))
            order = cp.argsort(phases, axis=1).astype(cp.int32)
            flux = cp.empty((rows, stride), dtype=cp.float32)
            errors = cp.empty_like(flux)
            invvar = cp.empty_like(flux)
            prep.get_function('patchData')(((stride + 255) // 256, rows), (256,),
                (flux, errors, stride_gpu, order, pad_gpu, y_gpu, dy_gpu, n_gpu))
            prep.get_function('calcInverseSquaredPatchedDy')(((stride + 255) // 256, rows), (256,),
                (invvar, errors, stride_gpu))
            edge = cp.empty(rows, dtype=cp.float32)
            prep.get_function('calcEdgeEffectCorrections')(((rows + 255) // 256,), (256,),
                (edge, flux, invvar, stride_gpu, pad_gpu, rows_gpu))
            if native_prefix:
                prefix = _native_flux_prefix(flux)
            else:
                prefix = cp.cumsum(flux, axis=1)
            base_error = cp.empty_like(flux)
            prep.get_function('calculate_base_error')(((stride + 255) // 256, rows), (256,),
                (base_error, flux, invvar, np.int32(stride), np.int32(rows)))
            error_prefix = cp.cumsum(base_error, axis=1)
            partial = cp.empty((rows, nt), dtype=cp.float32)
            keys = cp.empty((rows, nt), dtype=cp.uint64)
            depths = cp.empty((rows, nt), dtype=cp.float32)
            if full:
                fullsum = cp.empty((rows, nw), dtype=cp.float32)
                scan.get_function('tls_reference_fullsum_legacy')(((rows + 255) // 256,), (256,),
                    (flux, invvar, w_gpu, np.int32(rows), np.int32(stride), np.int32(nw), fullsum))
                delta = cp.empty((rows, nw, ndata), dtype=cp.float32)
                full_grid = ((ndata + 255) // 256, nw, rows)
                scan.get_function('tls_reference_ootr_delta')(full_grid, (256,),
                    (flux, invvar, w_gpu, np.int32(rows), np.int32(ndata),
                     np.int32(stride), np.int32(nw), delta))
                ootr = cp.cumsum(delta, axis=-1)
                scan.get_function('tls_reference_ootr_add')(full_grid, (256,),
                    (ootr, fullsum, np.int32(rows), np.int32(ndata), np.int32(nw)))
                scan.get_function('tls_reference_full_search')((nt, rows), (256,),
                    (flux, invvar, prefix, fullsum, ootr, edge, w_gpu, templates_gpu,
                     overshoot_gpu, tile_duration, tile_first, np.int32(rows),
                     np.int32(ndata), np.int32(stride), np.int32(nw),
                     np.int32(template_stride), np.int32(nt),
                     np.float32(transit_depth_min), partial, keys, depths))
            else:
                scan.get_function('tls_reference_search')((nt, rows), (256,),
                    (flux, invvar, prefix, error_prefix, edge, w_gpu, templates_gpu,
                     overshoot_gpu, tile_duration, tile_first, np.int32(rows),
                     np.int32(ndata), np.int32(stride), np.int32(nw),
                     np.int32(template_stride), np.int32(nt), np.int32(skip_factor),
                     np.float32(transit_depth_min), partial, keys, depths))
            out_chi2 = cp.empty(rows, dtype=cp.float32)
            out_start, out_index, out_width = (cp.empty(rows, dtype=cp.int32) for _ in range(3))
            out_depth = cp.empty(rows, dtype=cp.float32)
            scan.get_function('tls_reference_reduce')((rows,), (256,),
                (partial, keys, depths, w_gpu, np.int32(rows), np.int32(ndata),
                 np.int32(nt), out_chi2, out_start, out_index, out_width, out_depth))
            safe_start = cp.clip(out_start, 0, ndata - 1)
            original_start = t_gpu[order[cp.arange(rows, dtype=cp.int32), safe_start]]
            original_start = cp.where(out_start >= 0, original_start, np.nan)
            result['chi2'][start:stop] = out_chi2.get()
            result['start'][start:stop] = out_start.get()
            result['start_time'][start:stop] = original_start.get()
            local_index = out_index.get()
            result['width_index'][start:stop] = np.where(local_index >= 0, ids[np.maximum(local_index, 0)], -1)
            result['width'][start:stop] = out_width.get()
            result['depth'][start:stop] = out_depth.get()
            if capture:
                captured.append(dict(start=start, stop=stop, ids=ids,
                    phases=phases.get(), order=order.get(), flux=flux.get(),
                    invvar=invvar.get(), prefix=prefix.get(),
                    error_prefix=error_prefix.get(), edge=edge.get()))
    if capture:
        result['captured'] = captured
    cp.cuda.runtime.deviceSynchronize()
    return result


def _select_durations(selection, indices):
    if selection is None:
        return None
    return {key: selection[key][indices] for key in
            ('width_minima', 'width_maxima', 'requested_qmin', 'requested_qmax')}


def search_fast(t, y, dy, periods, *, group_size=None, work_chunk=256,
                T0_fit_margin=.125, duration_grid_step=1.1,
                oversampling_factor=3, u=None, limb_dark='quadratic',
                transit_template='default', template_parameters=None,
                qmin=None, qmax=None, n_durations=None,
                sde_kernel_size=None, **kwargs):
    prepared = reference.preprocess_inputs(t, y, dy)
    periods = np.asarray(periods, dtype=np.float64)
    order = np.argsort(periods, kind='stable')
    periods = periods[order]
    selection = None
    if qmin is not None or qmax is not None:
        if qmin is None or qmax is None:
            raise ValueError('provide both qmin and qmax')
        def aligned(value):
            value = np.broadcast_to(np.asarray(value, dtype=np.float64), periods.shape)
            return value[order]
        selection = reference.augment_duration_grid(
            periods, len(prepared['t']), aligned(qmin), aligned(qmax),
            duration_grid_step=duration_grid_step, n_durations=n_durations)
    cache = reference.build_cache(
        periods, len(prepared['t']), duration_grid_step=duration_grid_step,
        u=u, limb_dark=limb_dark, transit_template=transit_template,
        template_parameters=template_parameters,
        fractional_durations=None if selection is None else selection['fractional_durations'])
    skip = max(int(1 / T0_fit_margin), 8) if T0_fit_margin > 0 else 2**30
    raw = raw_search(periods, prepared['t'], prepared['y'], prepared['dy'], cache,
        group_size=group_size, work_chunk=work_chunk, skip_factor=skip,
        duration_selection=selection, **kwargs)
    spectra = reference.native_spectra(raw['chi2'], oversampling_factor,
                                       kernel_size=sde_kernel_size)
    index = spectra['primary_index']
    return dict(periods=periods, prepared=prepared, cache=cache, raw=raw,
                duration_selection=selection, spectra=spectra,
                period=None if index is None else periods[index])


def search_full(t, y, dy, periods, *, group_size=None, work_chunk=256,
                T0_fit_margin=.125, duration_grid_step=1.1,
                oversampling_factor=3, u=None, limb_dark='quadratic',
                transit_template='default', template_parameters=None,
                qmin=None, qmax=None, n_durations=None,
                sde_kernel_size=None, refine_top_k=None, **kwargs):
    """Full-stage native arithmetic, with finite first-stage candidates.

    Masked coarse periods are excluded before candidate ranking. Harmonics
    remain real trial periods and can legitimately replace a coarse mask
    when their no-skip search obtains a valid fit.
    """
    result = search_fast(t, y, dy, periods, group_size=group_size,
        work_chunk=work_chunk, T0_fit_margin=T0_fit_margin,
        duration_grid_step=duration_grid_step, oversampling_factor=oversampling_factor,
        u=u, limb_dark=limb_dark, transit_template=transit_template,
        template_parameters=template_parameters, qmin=qmin, qmax=qmax,
        n_durations=n_durations, sde_kernel_size=sde_kernel_size, **kwargs)
    if result['period'] is None:
        return result
    p, prepared, cache = result['periods'], result['prepared'], result['cache']
    initial_mask = np.ma.getmaskarray(result['spectra']['chi2'])
    masked_periods = np.ma.array(p, mask=initial_mask)
    chi2 = np.ma.array(result['raw']['chi2'], mask=initial_mask, copy=True)
    initial_power = result['spectra']['power'].copy()
    spectra_history = [result['spectra']]
    candidates = reference.refinement_candidate_indices(masked_periods, initial_power)
    if refine_top_k is not None:
        candidates = candidates[:refine_top_k]
    if not len(candidates):
        return result
    group = result['raw']['group_size']
    selection = result['duration_selection']
    refined = raw_search(p[candidates], prepared['t'], prepared['y'], prepared['dy'], cache,
        group_size=group, work_chunk=work_chunk, full=True,
        duration_selection=_select_durations(selection, candidates), **kwargs)
    chi2[candidates] = refined['chi2']
    spectra = reference.native_spectra(chi2, oversampling_factor, mask_outliers=False,
                                       kernel_size=sde_kernel_size)
    spectra_history.append(spectra)
    primary = int(candidates[np.argmax(spectra['power'][candidates])])
    harmonic = reference.harmonic_candidate_indices(masked_periods, p[primary])
    harmonic_results = raw_search(p[harmonic], prepared['t'], prepared['y'], prepared['dy'], cache,
        group_size=group, work_chunk=work_chunk, full=True,
        duration_selection=_select_durations(selection, harmonic), **kwargs)
    chi2[harmonic] = harmonic_results['chi2']
    spectra = reference.native_spectra(chi2, oversampling_factor, mask_outliers=False,
                                       kernel_size=sde_kernel_size)
    spectra_history.append(spectra)
    primary = int(harmonic[np.argmax(spectra['power'][harmonic])])
    final = raw_search(p[primary:primary+1], prepared['t'], prepared['y'], prepared['dy'], cache,
        group_size=1, work_chunk=1, full=True,
        duration_selection=_select_durations(selection, slice(primary, primary+1)), **kwargs)
    result.update(coarse_raw=result['raw'], coarse_spectra=result['spectra'], spectra=spectra,
        spectra_history=spectra_history,
        coarse_period=result['period'], period=float(p[primary]), primary_index=primary,
        candidates=candidates, refined=refined, harmonics=harmonic,
        harmonic_results=harmonic_results, final=final)
    return result
