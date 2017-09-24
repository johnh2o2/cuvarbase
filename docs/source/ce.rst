Conditional Entropy
===================

The conditional entropy period finder [G2013]_ phase-folds the data at each trial frequencies and estimates
the conditional entropy :math:`H(m|\phi)` of the data. 

Here,

:math:`H(m|\phi) = H(m, \phi) - H(\phi) = \sum_{m,\phi}p(m, \phi)\log\left(\frac{p(\phi)}{p(m, \phi)}\right)`

where :math:`p(m, \phi)` is the density of points that fall within the bin located at phase :math:`\phi` and magnitude :math:`m` and `p(\phi)` is the density of points that fall within the range :math:`\phi \pm \delta \phi`, where :math:`\delta\phi` is the width of the bin located at phase :math:`\phi`.

.. [G2013] `Graham et al. 2013 <http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1306.6664>`_
