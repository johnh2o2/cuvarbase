use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

mod aov;
mod basicstats;
mod bls;
mod ce;
mod dmdt;
mod fold;
mod fourier;
mod fpw;
mod highcadence;
mod ls;
mod mf;
mod mhf;
mod peaks;
mod vn;

// ===========================================================================
// Full-periodogram functions (existing)
// ===========================================================================

/// Compute batched Conditional Entropy periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_ce_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = ce::calc_ce(
                    times_vecs[ci],
                    mags_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_phase,
                    num_mag,
                    phase_overlap,
                    mag_overlap,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_ce_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute batched Analysis of Variance periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_aov_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    num_overlap: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = aov::calc_aov(
                    times_vecs[ci],
                    mags_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_bins,
                    num_overlap,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_aov_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute batched Lomb-Scargle periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_ls_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result =
                    ls::calc_ls(times_vecs[ci], mags_vecs[ci], periods_slice, period_dts_slice);
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_ls_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute batched FPW periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_fpw_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = fpw::calc_fpw(
                    times_vecs[ci],
                    mags_vecs[ci],
                    errs_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_bins,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_fpw_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute batched Box Least Squares periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_bls_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    qmin: f32,
    qmax: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = bls::calc_bls(
                    times_vecs[ci],
                    mags_vecs[ci],
                    errs_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_bins,
                    qmin,
                    qmax,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_bls_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute batched Fourier decomposition (feature extraction).
///
/// Returns a 2D numpy array of shape (n_curves, 14).
/// Features: [power, BIC, offset, slope, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5]
#[pyfunction]
fn calc_fourier_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let periods_slice = periods.as_slice()?;
    let n_curves = times_list.len();

    // Extract slices while the GIL is held
    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    // Release the GIL for the Rayon-parallel computation
    let flat = py.allow_threads(|| {
        fourier::calc_fourier_batch(&times_vecs, &mags_vecs, &errs_vecs, periods_slice)
    });

    // Reshape into a 2D array
    let n_feat = fourier::NUM_FEATURES;
    let mut output = Array2::<f32>::zeros((n_curves, n_feat));
    for i in 0..n_curves {
        for j in 0..n_feat {
            output[[i, j]] = flat[i * n_feat + j];
        }
    }

    Ok(output.into_pyarray(py).into())
}

// ===========================================================================
// Viterbi Narrowband
// ===========================================================================

/// Compute batched Viterbi Narrowband periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
#[pyfunction]
fn calc_vn_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
    bandwidth: usize,
    margin: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = vn::calc_vn(
                    times_vecs[ci],
                    mags_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_phase,
                    num_mag,
                    phase_overlap,
                    mag_overlap,
                    bandwidth,
                    margin,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_vn_batched");
    Ok(output.into_pyarray(py).into())
}

/// Fused VN + peak finding.  Returns (peak_indices, peak_values) each of
/// shape (n_curves, n_peaks).
#[pyfunction]
fn calc_vn_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
    bandwidth: usize,
    margin: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let flat = vn::calc_vn(
            ts,
            ms,
            periods_s,
            period_dts_s,
            num_phase,
            num_mag,
            phase_overlap,
            mag_overlap,
            bandwidth,
            margin,
        );
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

// ===========================================================================
// Peak-finding functions
// ===========================================================================

/// Find top-N peaks in pre-computed periodograms.
///
/// Takes a 3D array of shape (n_curves, n_periods, n_pdts) and returns two
/// 2D arrays: peak_indices (n_curves, n_peaks) and peak_values (n_curves, n_peaks),
/// both sorted best-first per curve.
///
/// Peaks are local extrema (maxima if use_max=True, minima otherwise)
/// separated by at least `min_distance` samples in the flattened
/// (n_periods × n_pdts) scan order.
#[pyfunction]
fn find_top_peaks_batched<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f32>,
    n_peaks: usize,
    min_distance: usize,
    use_max: bool,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let shape = data.shape();
    let n_curves = shape[0];
    let n_periods = shape[1];
    let n_pdts = shape[2];
    let flat_len = n_periods * n_pdts;

    let mut out_indices = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_values = Array2::<f32>::zeros((n_curves, n_peaks));

    let data_arr = data.as_array();

    for curve_idx in 0..n_curves {
        // Flatten the 2D periodogram for this curve into a contiguous slice
        let mut flat = Vec::with_capacity(flat_len);
        for p in 0..n_periods {
            for d in 0..n_pdts {
                flat.push(data_arr[[curve_idx, p, d]]);
            }
        }

        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, use_max);

        for i in 0..idx.len() {
            out_indices[[curve_idx, i]] = idx[i];
            out_values[[curve_idx, i]] = val[i];
        }
    }

    Ok((
        out_indices.into_pyarray(py).into(),
        out_values.into_pyarray(py).into(),
    ))
}

// ---------------------------------------------------------------------------
// Fused: compute periodogram + find peaks without materialising full output.
//
// Memory: O(n_periods × n_pdts) per curve (temporary) instead of
//         O(n_curves × n_periods × n_pdts) for the full 3D array.
// ---------------------------------------------------------------------------

/// Fused CE + peak finding.  Returns (peak_indices, peak_values) each of
/// shape (n_curves, n_peaks).
#[pyfunction]
fn calc_ce_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_phase: usize,
    num_mag: usize,
    phase_overlap: usize,
    mag_overlap: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let flat = ce::calc_ce(
            ts,
            ms,
            periods_s,
            period_dts_s,
            num_phase,
            num_mag,
            phase_overlap,
            mag_overlap,
        );
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, false);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

/// Fused AOV + peak finding.
#[pyfunction]
fn calc_aov_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    num_overlap: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let flat = aov::calc_aov(ts, ms, periods_s, period_dts_s, num_bins, num_overlap);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

/// Fused LS + peak finding.
#[pyfunction]
fn calc_ls_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let flat = ls::calc_ls(ts, ms, periods_s, period_dts_s);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

/// Fused FPW + peak finding.
#[pyfunction]
fn calc_fpw_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let es = errs_list[ci].as_slice()?;
        let flat = fpw::calc_fpw(ts, ms, es, periods_s, period_dts_s, num_bins);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

/// Fused BLS + peak finding.
#[pyfunction]
fn calc_bls_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    qmin: f32,
    qmax: f32,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let es = errs_list[ci].as_slice()?;
        let flat = bls::calc_bls(ts, ms, es, periods_s, period_dts_s, num_bins, qmin, qmax);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

// ===========================================================================
// Matched Filter periodogram + features
// ===========================================================================

/// Compute batched Matched Filter periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
/// Each value is the combined score (max_corr × R² × coverage).
#[pyfunction]
fn calc_mf_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = mf::calc_mf(
                    times_vecs[ci],
                    mags_vecs[ci],
                    errs_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    num_bins,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_mf_batched");
    Ok(output.into_pyarray(py).into())
}

/// Fused MF + peak finding.
#[pyfunction]
fn calc_mf_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let es = errs_list[ci].as_slice()?;
        let flat = mf::calc_mf(ts, ms, es, periods_s, period_dts_s, num_bins);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

/// Compute batched Matched Filter feature extraction.
///
/// Returns a 2D numpy array of shape (n_curves, 7).
/// Features: [best_sawtooth, best_sinusoidal, best_eclipsing, R², amp_snr, n_filled, combined]
#[pyfunction]
fn calc_mf_features_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    num_bins: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let periods_slice = periods.as_slice()?;
    let n_curves = times_list.len();

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        mf::calc_mf_features_batch(&times_vecs, &mags_vecs, &errs_vecs, periods_slice, num_bins)
    });

    let n_feat = mf::NUM_MF_FEATURES;
    let mut output = Array2::<f32>::zeros((n_curves, n_feat));
    for i in 0..n_curves {
        for j in 0..n_feat {
            output[[i, j]] = flat[i * n_feat + j];
        }
    }

    Ok(output.into_pyarray(py).into())
}

// ===========================================================================
// Multi-Harmonic Fourier periodogram
// ===========================================================================

/// Compute batched Multi-Harmonic Fourier periodograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_periods, n_pdts).
/// Each value is ΔBIC = BIC_flat - BIC_best (higher = more periodic).
#[pyfunction]
fn calc_mhf_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    max_harmonics: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let n_periods = periods_slice.len();
    let n_pdts = period_dts_slice.len();
    let per_curve = n_periods * n_pdts;

    if per_curve == 0 {
        let output = Array3::<f32>::zeros((n_curves, n_periods, n_pdts));
        return Ok(output.into_pyarray(py).into());
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * per_curve];
        output
            .par_chunks_mut(per_curve)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = mhf::calc_mhf(
                    times_vecs[ci],
                    mags_vecs[ci],
                    errs_vecs[ci],
                    periods_slice,
                    period_dts_slice,
                    max_harmonics,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array3::from_shape_vec((n_curves, n_periods, n_pdts), flat)
        .expect("shape mismatch in calc_mhf_batched");
    Ok(output.into_pyarray(py).into())
}

/// Compute per-K MHF ΔBIC at a single period per curve.
///
/// Returns a 2D numpy array of shape (n_curves, max_harmonics + 2).
/// Each row is [ΔBIC_k0, ΔBIC_k1, ..., ΔBIC_kN, best_k].
#[pyfunction]
fn calc_mhf_per_k_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    max_harmonics: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let periods_slice = periods.as_slice()?;
    let period_dts_slice = period_dts.as_slice()?;
    let n_curves = times_list.len();
    let max_k = max_harmonics.min(5);
    let out_cols = max_k + 2; // ΔBIC for k=0..max_k, plus best_k

    if n_curves != periods_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods must have one entry per curve",
        ));
    }
    if n_curves != period_dts_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "period_dts must have one entry per curve",
        ));
    }

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        let mut output = vec![0.0f32; n_curves * out_cols];
        output
            .par_chunks_mut(out_cols)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let result = mhf::calc_mhf_per_k(
                    times_vecs[ci],
                    mags_vecs[ci],
                    errs_vecs[ci],
                    periods_slice[ci],
                    period_dts_slice[ci],
                    max_harmonics,
                );
                chunk.copy_from_slice(&result);
            });
        output
    });

    let output = Array2::from_shape_vec((n_curves, out_cols), flat)
        .expect("shape mismatch in calc_mhf_per_k_batched");
    Ok(output.into_pyarray(py).into())
}

/// Fused MHF + peak finding.
#[pyfunction]
fn calc_mhf_peaks_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    periods: PyReadonlyArray1<'py, f32>,
    period_dts: PyReadonlyArray1<'py, f32>,
    max_harmonics: usize,
    n_peaks: usize,
    min_distance: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let n_curves = times_list.len();
    let periods_s = periods.as_slice()?;
    let period_dts_s = period_dts.as_slice()?;

    let mut out_idx = Array2::<i64>::from_elem((n_curves, n_peaks), -1);
    let mut out_val = Array2::<f32>::zeros((n_curves, n_peaks));

    for ci in 0..n_curves {
        let ts = times_list[ci].as_slice()?;
        let ms = mags_list[ci].as_slice()?;
        let es = errs_list[ci].as_slice()?;
        let flat = mhf::calc_mhf(ts, ms, es, periods_s, period_dts_s, max_harmonics);
        let (idx, val) = peaks::find_top_peaks(&flat, n_peaks, min_distance, true);
        for i in 0..idx.len() {
            out_idx[[ci, i]] = idx[i];
            out_val[[ci, i]] = val[i];
        }
    }

    Ok((
        out_idx.into_pyarray(py).into(),
        out_val.into_pyarray(py).into(),
    ))
}

// ===========================================================================
// High-cadence removal
// ===========================================================================

/// Batch high-cadence removal across multiple light curves.
///
/// Returns a list of (times, mags, errs) tuples, each filtered to keep only
/// points separated by at least `cadence_minutes` minutes.
#[pyfunction]
fn remove_high_cadence_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
    cadence_minutes: f32,
) -> PyResult<Vec<(Py<PyArray1<f32>>, Py<PyArray1<f32>>, Py<PyArray1<f32>>)>> {
    let cadence_days = cadence_minutes / 1440.0;

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let results = py.allow_threads(|| {
        highcadence::remove_high_cadence_batch(&times_vecs, &mags_vecs, &errs_vecs, cadence_days)
    });

    Ok(results
        .into_iter()
        .map(|(t, m, e)| {
            (
                PyArray1::from_vec(py, t).into(),
                PyArray1::from_vec(py, m).into(),
                PyArray1::from_vec(py, e).into(),
            )
        })
        .collect())
}

// ===========================================================================
// dm-dt histograms
// ===========================================================================

/// Compute batched dm-dt histograms.
///
/// Returns a 3D numpy array of shape (n_curves, n_dm_bins, n_dt_bins),
/// L2-normalised per curve.
#[pyfunction]
fn compute_dmdt_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    dt_edges: PyReadonlyArray1<'py, f32>,
    dm_edges: PyReadonlyArray1<'py, f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let dt_edges_s = dt_edges.as_slice()?;
    let dm_edges_s = dm_edges.as_slice()?;
    let n_curves = times_list.len();
    let n_dt_bins = if dt_edges_s.len() > 1 {
        dt_edges_s.len() - 1
    } else {
        0
    };
    let n_dm_bins = if dm_edges_s.len() > 1 {
        dm_edges_s.len() - 1
    } else {
        0
    };

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py.allow_threads(|| {
        dmdt::compute_dmdt_batch(&times_vecs, &mags_vecs, dt_edges_s, dm_edges_s)
    });

    let mut output = Array3::<f32>::zeros((n_curves, n_dm_bins, n_dt_bins));
    for ci in 0..n_curves {
        let offset = ci * n_dm_bins * n_dt_bins;
        for dm in 0..n_dm_bins {
            for dt in 0..n_dt_bins {
                output[[ci, dm, dt]] = flat[offset + dm * n_dt_bins + dt];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

// ===========================================================================
// Basic statistics
// ===========================================================================

/// Compute batched basic light curve statistics.
///
/// Returns a 2D numpy array of shape (n_curves, 22).
#[pyfunction]
fn calc_basic_stats_batched<'py>(
    py: Python<'py>,
    times_list: Vec<PyReadonlyArray1<'py, f32>>,
    mags_list: Vec<PyReadonlyArray1<'py, f32>>,
    errs_list: Vec<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Py<PyArray2<f32>>> {
    let n_curves = times_list.len();

    let times_vecs: Vec<&[f32]> = times_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let mags_vecs: Vec<&[f32]> = mags_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;
    let errs_vecs: Vec<&[f32]> = errs_list
        .iter()
        .map(|a| a.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let flat = py
        .allow_threads(|| basicstats::calc_basic_stats_batch(&times_vecs, &mags_vecs, &errs_vecs));

    let n_feat = basicstats::NUM_BASIC_STATS;
    let mut output = Array2::<f32>::zeros((n_curves, n_feat));
    for i in 0..n_curves {
        for j in 0..n_feat {
            output[[i, j]] = flat[i * n_feat + j];
        }
    }

    Ok(output.into_pyarray(py).into())
}

// ===========================================================================
// Module registration
// ===========================================================================

/// Native CPU implementations of period-finding algorithms.
#[pymodule]
fn periodfind_cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Full periodogram
    m.add_function(wrap_pyfunction!(calc_ce_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_aov_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ls_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fpw_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_bls_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fourier_batched, m)?)?;
    // Peak finding
    m.add_function(wrap_pyfunction!(find_top_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ce_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_aov_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ls_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fpw_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_bls_peaks_batched, m)?)?;
    // Matched filter
    m.add_function(wrap_pyfunction!(calc_mf_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_mf_peaks_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_mf_features_batched, m)?)?;
    // Multi-Harmonic Fourier
    m.add_function(wrap_pyfunction!(calc_mhf_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_mhf_per_k_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_mhf_peaks_batched, m)?)?;
    // Viterbi Narrowband
    m.add_function(wrap_pyfunction!(calc_vn_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_vn_peaks_batched, m)?)?;
    // Feature extraction
    m.add_function(wrap_pyfunction!(remove_high_cadence_batched, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dmdt_batched, m)?)?;
    m.add_function(wrap_pyfunction!(calc_basic_stats_batched, m)?)?;
    Ok(())
}
