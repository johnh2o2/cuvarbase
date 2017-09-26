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


.. [BLS] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_