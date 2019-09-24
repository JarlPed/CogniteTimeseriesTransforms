# CogniteTimeseriesTransforms
This is a simple project to demonstrate the capabilities of simple discrete time series transforms for the Valhall field data (public data from Cognite)

The method used for this project is a coupled Simpsons-Aitken integration method with irregular time series spacing. The Aitken's iteration extrapolate the integral to infinite (at least by its methology) to aid the numerical Laplace transform.

Three files are provided; utils.py contain the core functions for transforming data series (other integrands can easily be customized if wished). Testbed_DiscreteLaplace.py contains a test to benchmark the theoretical deviation of the numerical tools as the time series are derived symbolically. Cognite_Implementation.py contains a simple database scraping of the AkerBP-Valhalla asset and numerical analysis for a single time series.