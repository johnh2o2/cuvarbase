Box least squares (BLS) periodogram
***********************************

The box-least squares periodogram [BLS]_ searches for the periodic dips in brightness that occur when, e.g., a planet passes in front of its host star. The algorithm fits
a `boxcar function <https://en.wikipedia.org/wiki/Boxcar_function>`_ to the data. The parameters used are

- ``q``: the transit duration as a fraction of the period :math:`t_{\rm trans} / P`
- ``phi0``: the phase offset of the transit (from 0)
- ``delta``: the difference between the out-of-transit brightness and the brightness during transit 
- ``y0``: The out-of-transit brightness


.. plot:: plots/bls_transit_diagram.py


Using ``cuvarbase`` BLS
-----------------------


.. plot:: plots/bls_example.py
	:include-source:


A shortcut: assuming orbital mechanics
--------------------------------------

If you assume :math:`R_p\ll R_{\star}`, :math:`M_p\ll M_{\star}`, :math:`L_p\ll L_{\star}`, and :math:`e\ll 1`,  where :math:`e` is the ellipticity of the planetary orbit, :math:`L` is the luminosity, :math:`R` is the radius, and :math:`M` mass, you can eliminate a free parameter.

This is because the orbital period obeys `Kepler's third law <https://en.wikipedia.org/wiki/Kepler's_laws_of_planetary_motion#Third_law>`_,

.. math::
	P^2 \approx \frac{4\pi^2a^3}{G(M_p + M_{\star})}

.. plot:: plots/planet_transit_diagram.py


The angle of the transit is

.. math::

	\theta = 2{\rm arcsin}\left(\frac{R_p + R_{\star}}{a}\right)

and :math:`q` is therefore :math:`\theta / (2\pi)`. Thus we have a relation between :math:`q` and the period :math:`P`

.. math::

	\sin{\pi q} = (R_p + R_{\star})\left(\frac{4\pi^2}{P^2 G(M_p + M_{\star})}\right)^{1/3}

By incorporating the fact that

.. math::
	
	R_{\star} = \left(\frac{3}{4\pi\rho_{\star}}\right)^{1/3}M_{\star}^{1/3}

where :math:`\rho_{\star}` is the average stellar density of the host star, we can write

.. math::

	\sin{\pi q} = \frac{(1 + r)}{(1 + m)^{1/3}} \left(\frac{3\pi}{G\rho_{\star}}\right)^{1/3} P^{-2/3}

where :math:`r = R_p / R_{\star}` and :math:`m = M_p / M_{\star}`. We can get rid of the constant factors and convert this to more intuitive units to obtain

.. math::

	\sin{\pi q} \approx 0.238 (1 + r - \frac{m}{3} + \dots{}) \left(\frac{\rho_{\star}}{\rho_{\odot}}\right)^{-1/3} \left(\frac{P}{\rm day}\right)^{-2/3}

where here we've expanded :math:`(1 + r) / (1 + m)^{1/3}` to first order in :math:`r` and :math:`m`.


Using the Keplerian assumption in ``cuvarbase``
-----------------------------------------------

.. plot:: plots/bls_example_transit.py
	:include-source:


Period spacing considerations
-----------------------------

The frequency spacing :math:`\delta f` needed to resolve a BLS signal with width :math:`q`, is

.. math::
	\delta f \lesssim \frac{q}{T}

where :math:`T` is the baseline of the observations (:math:`T = {\rm max}(t) - {\rm min}(t)`). This can be especially problematic if no assumptions are made about the nature of the signal (e.g., a Keplerian assumption). If you want to resolve a transit signal with a few observations, the minimum :math:`q` value that you would need to search is :math:`\propto 1/N` where :math:`N` is the number of observations.

For a typical Lomb-Scargle periodogram, the frequency spacing is :math:`\delta f \lesssim 1/T`, so running a BLS spectrum with an adequate frequency spacing over the same frequency range requires a factor of :math:`\mathcal{O}(N)` more trial frequencies, each of which requiring :math:`\mathcal{O}(N)` computations to estimate the best fit BLS parameters. That means that BLS scales as :math:`\mathcal{O}(N^2N_f)` while Lomb-Scargle only scales as :math:`\mathcal{O}(N_f\log N_f)`

However, if you can use the assumption that the transit is caused by an edge-on transit of a circularly orbiting planet, we not only eliminate a degree of freedom, but (assuming :math:`\sin{\pi q}\approx \pi q`)

.. math::
	
	\delta f \propto q \propto f^{2/3}

The minimum frequency you could hope to measure a transit period would be :math:`f_{\rm min} \approx 2/T`, and the maximum frequency is determined by :math:`\sin{\pi q} < 1` which implies

.. math::

	f_{max} = 8.612~{\rm c/day}~\times \left(1 - \frac{3r}{2} + \frac{m}{2} -\dots{}\right) \sqrt{\frac{\rho_{\star}}{\rho_{\odot}}}


For a 10 year baseline, this translates to :math:`2.7\times 10^5` trial frequencies. The number of trial frequencies needed to perform Lomb-Scargle over this frequency range is only about :math:`3.1\times 10^4`, so 8-10 times less. However, if we were to search the *entire* range of possible :math:`q` values at each trial frequency instead of making a Keplerian assumption, we would instead require :math:`5.35\times 10^8` trial frequencies, so the Keplerian assumption reduces the number of frequencies by over 1,000.


Sparse BLS for small datasets
------------------------------

For datasets with a small number of observations, the standard BLS algorithm that bins observations and searches over a grid of transit parameters can be inefficient. The "Sparse BLS" algorithm [SparseBLS]_ avoids this redundancy by directly testing all pairs of observations as potential transit boundaries.

At each trial frequency, the observations are sorted by phase. Then, instead of searching over a grid of (phase, duration) parameters, the algorithm considers each pair of consecutive observations (i, j) as defining:

- Transit start phase: :math:`\phi_0 = \phi_i`
- Transit duration: :math:`q = \phi_j - \phi_i`

This approach has complexity :math:`\mathcal{O}(N_{\rm freq} \times N_{\rm data}^2)` compared to :math:`\mathcal{O}(N_{\rm freq} \times N_{\rm data} \times N_{\rm bins})` for the standard gridded approach. For small datasets (typically :math:`N_{\rm data} < 500`), sparse BLS can be more efficient as it avoids testing redundant parameter combinations.

Using Sparse BLS in ``cuvarbase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``eebls_transit`` function automatically selects between sparse BLS (for small datasets) and the GPU-accelerated standard BLS (for larger datasets):

.. code-block:: python

    from cuvarbase.bls import eebls_transit
    import numpy as np
    
    # Generate small dataset (e.g., 100 observations)
    t = np.sort(np.random.rand(100)) * 365  # 1 year baseline
    # ... (generate y, dy from your data)
    
    # Automatically uses sparse BLS for ndata < 500
    freqs, powers, solutions = eebls_transit(
        t, y, dy,
        fmin=0.1,  # minimum frequency
        fmax=10.0  # maximum frequency
    )
    
    # Or explicitly control the method:
    freqs, powers, solutions = eebls_transit(
        t, y, dy,
        fmin=0.1, fmax=10.0,
        use_sparse=True  # Force sparse BLS
    )

You can also use sparse BLS directly with ``sparse_bls_cpu``:

.. code-block:: python

    from cuvarbase.bls import sparse_bls_cpu
    
    # Define trial frequencies
    freqs = np.linspace(0.1, 10.0, 1000)
    
    # Run sparse BLS
    powers, solutions = sparse_bls_cpu(t, y, dy, freqs)
    
    # solutions is a list of (q, phi0) tuples for each frequency
    best_idx = np.argmax(powers)
    best_freq = freqs[best_idx]
    best_q, best_phi0 = solutions[best_idx]


.. [BLS] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_
.. [SparseBLS] `Burdge et al. 2021 <https://arxiv.org/abs/2103.06193>`_