# Rider Control IDentification (RCID)

This package contains reusable functions for data processing, rider-control identification, modeling of behavioral parameters and testing
within the scope of the publication *Stochastic Control Behavior of the Balancing Rider for Cycling Safety in Traffic Simulation*.

## Modules

`constant`: A module containing global constants. Added late and not used universally.

`data_processing`: A module for processign the raw data and loading/writing processed data. 

`evaluation`: A module for evaluating control identification results.

`identification`: A module for identifying rider control parameters from step response measurements.

`path_manager`: A module for managing paths to individual elements of data and results.

`pole_modelling`: A module for modeling behavioral distributions, creating random samples of behavioral parameters and testing the predicted trajectory distributions.

`simulation`: A module for simulating different cycling scenarios within the scope of the publication.

`ukf`: A module implementing an Unscented Kalman Filter for fusing raw data from different sensors.

`utils`: Helper functions

`visualisation`: Vizualisation functions

## Installation

1. Install dependencies. This package depends on a few custom packages that are not available on PyPi. Navigate to their public repositories on GitHub and follow installation instructions there. Make sure to install the correct version specified in the `pyproject.toml`.
   
   - `mypyutils` [GitHub](https://github.com/chris-konrad/mypyutils)
   - `pypaperutils` [GitHub](https://github.com/chris-konrad/pypaperutils)
   - `pytrafficutils` [GitHub](https://github.com/chris-konrad/pytrafficutils)
   - `trajdatamanager` [GitHub](https://github.com/chris-konrad/trajdatamanager)
   - `cyclistsocialforce` [GitHub](https://github.com/chris-konrad/cyclistsocialforce)

2. Install the package by navigating to its directory and calling
   
   ```
   pip install . 
   ```

## Usage and Documentation

This package does not come with an API documentation. Please refer to the docstrings wherever available. The analysis scripts in `/code/scripts` serve as examples
how to use the funtions and classes of this package.

## License

This package is licensed under the terms of the [MIT license](LICENSE).

## Citation

If you use this package in your research, please cite our corresponding publication:

Konrad, C. M., Happee, R., Moore, J. K., & Dabiri, A. (2025). *Stochastic Control Behavior of the Balancing Rider for Cycling Safety in Traffic Simulation* [Manuscript].
