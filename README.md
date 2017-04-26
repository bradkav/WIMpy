# WIMpy
[**Work in progress**] Python code for doing fits to (mock) dark matter direct detection data.

Details of how to use the code will be added soon!

## The WIMpy Code

The WIMpy code uses...

## Reproducing results and plots

#### Performing fits

The python script `WIM.py` is used to calculate the significance achievable with a particular set of experiments and a particular set of input Dark Matter parameters. It is called as:

`python WIM.py ENSEMBLE MASS INDEX DIR`.

Here, `ENSEMBLE` specifies which experimental ensemble to use `A`, `B`, `C` or `D`. `MASS` specifies the input DM mass in GeV. `INDEX` specifies the input DM couplings (in particular, in indexes the points on a grid of couplings).

`WIMpy` generates 50 mock data sets based on the input parameter points. For each data set, it uses a grid-refinement method to calculate the maximum likelihood and therefore the significance for discrimination between Dirac-like and Majorana-like couplings. The significances are output to a file named `Results_pINDEX.txt` in the relative path `DIR`.

#### Checking the likelihood calculator

`CompareNgrid.py`

#### Plotting

The key plots in the paper () are produced by running the python script `PlotContours.py`, which is called with:

`python PlotContours.py ENSEMBLE MASS`.

The python scripts runs through the results in the `../results/AP_ExptENSEMBLE_MASS/` directory, calculating the median significance and plotting the appropriate contours in the (f, c_n/c_p) plane. To generate all the contour plots from the data, simply run:

`./GenerateAllPlots.sh`

in the root directory, which generates all the figures which appear in the paper. 
