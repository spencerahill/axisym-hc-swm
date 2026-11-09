# Sobel-Schneider Model

Single layer model based on Sobel and Schneider 2009 and 2013. 

The model runs on an equatorial beta plane by default. `--geometry sphere` runs it on the sphere instead, from pole to pole in the coordinate y = a sin(latitude), with `--radius`, `--rot-rate`, `--lat-0` (the latitude of the forcing's heating maximum) and `--emfd-form` (the spherical form of the eddy momentum flux divergence); see `SCIENCE.md` §3.7.

# Reference

Sobel, A. H., and Schneider, T. (2009), Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing, J. Adv. Model. Earth Syst., 1, 10, doi:10.3894/JAMES.2009.1.10.

Sobel, A. H., and Schneider, T. (2013), Correction to “Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing”, J. Adv. Model. Earth Syst., 5, 654– 657, doi:10.1002/jame.20030.
