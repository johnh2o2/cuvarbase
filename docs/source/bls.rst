Box least squares (BLS) periodogram
===================================

The box-least squares periodogram [BLS]_ searches for the periodic dips in brightness that occur when, e.g., a planet passes in front of its host star. The algorithm fits
a `boxcar function <https://en.wikipedia.org/wiki/Boxcar_function>`_ to the data. The parameters used are

- ``q``: the transit duration as a fraction of the period :math:`t_{\rm trans} / P`
- ``phi0``: the phase offset of the transit (from 0)
- ``delta``: the difference between the out-of-transit brightness and the brightness during transit 
- ``y0``: The out-of-transit brightness


.. plot:: plots/bls.py


A shortcut: assuming orbital mechanics
--------------------------------------

If you assume :math:`R_p\ll R_{\star}`

Period spacing considerations
-----------------------------

The frequency spacing :math:`\delta f` needed to resolve a BLS signal with width :math:`q`, is

.. math::
	\delta f = \frac{q}{T}

where :math:`T` is the baseline of the observations (:math:`T = {\rm max}(t) - {\rm min}(t)`)


.. [BLS] `Kovacs et al. 2002 <http://adsabs.harvard.edu/abs/2002A%26A...391..369K>`_