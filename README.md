# CogniteTimeseriesTransforms
This is a simple project to demonstrate the capabilities of simple discrete time series transforms for the Valhall field data (public data from Cognite)

As of now, I have deviced a test to querry the assets and the associated data series. The time analysis consits of a Bode plot, and a Nyquist plot, however some more analysis regarding noize is to be done.

Noize (1 instance) is found to not be random by serial correlation tests showing that lag -> 20 steps is highly correlated ranging from 100 to 20 % correlation. Due to high sample batching at the instrumentation, the true measurement is taken as an average of several sampling points.

In Process: Aitken iteration is implemented to avoid large approximation errors at large omega's
