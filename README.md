# xSAGA: Extending the SAGA Survey

<img align="center" src="assets/ngc3044.png"/>  

## Identifying low-redshift galaxies with CNNs

We train a convolutional neural network (CNN) to distinguish low-redshift (*z* < 0.03) galaxies from the more numerous high-redshift objects. Each galaxy is given a CNN prediction (*p*<sub>CNN</sub>) betwen 0 and 1 that encodes how likely its redshift is *z* < 0.03. 

<p float="center">
  <img align="center" src="assets/saga-examples.png" width=480px alt="Examples image cutouts of xSAGA galaxies. Part of Figure 1 from the xSAGA I paper."/>
  <img align="center" src="assets/xsaga-examples.png" width=480px alt="Examples image cutouts of xSAGA galaxies. Part of Figure 1 from the xSAGA I paper."/> 
</p>

## Data

We feed image cutouts [DESI Legacy Imaging Surveys](https://www.legacysurvey.org/) as inputs to our CNN. Using the Legacy Survey RESTful API, we download *grz* (three-channel), 224×224 pixel images in `JPG` format. Example download scripts can be found in the `src/` directory.

Redshifts from the [Satellites Around Galactic Analogs (SAGA) Survey](https://ui.adsabs.harvard.edu/abs/2021ApJ...907...85M/abstract) comprise the training set labels. While the SAGA Stage II redshifts are publicly available [online](https://sagasurvey.org/), we use a proprietary redshift catalog that will be released in a future SAGA paper. 

After we have trained the CNN, we can make predictions for a test data set of 4.4 million photometrically selected candidates in SDSS/Legacy Survey (see paper for details). The CNN identifies over 100,000 galaxies as low-redshift candidates

## Results

We define CNN-selected galaxies as satellite galaxies if they are within 300 projected kpc from a spectroscopically confirmed *z* < 0.03 host galaxy (from [NASA-Sloan Atlas](https://www.sdss.org/dr16/manga/manga-target-selection/nsa/) version `1_0_1`). We compute the number of satellites around each host galaxy as a function of projected radius, while correcting for incompleteness, false positives, and non-satellite galaxy contamination (each of which is independently determined via cross-validation).

**More massive host galaxies have more satellites.** We see that host galaxies with higher stellar masses also tend to have more satellites (see left panel of figure below). This is in line with expectations, but it is exciting to see such a strong signal!

<img align="center" src="assets/profiles.png" alt="Satellite radial profiles (un-normalized on the left, and normalized on the right) in bins of host galaxy stellar mass. Figure 5 from the xSAGA I paper."/>  

**The radial distributions of satellites do not vary with host properties.** In the right panel above, we see that the normalized radial distributions of CNN-selected satellites are surprisingly insensitive to their host galaxies' stellar masses. The shapes of the radial profiles also do not appear to vary with the host's morphology or the magnitude gap between the brightest satellite and host galaxy.

**Satellites are most abundant around massive ellipticals.** In addition to the stellar mass dependence, we also find a modest increase in satellite richness around elliptical hosts (hatched shading in figure below) compared to around disky hosts (solid shading). The impact of morphology is stronger for more massive host galaxies.

<img align="center" width=480px src="assets/morphology.png" alt="Satellite radial profiles as a function of host stellar mass, separated into disky and elliptical morphologies. Figure 7 from the xSAGA I paper."/>  

**The satellite abundance correlates strongly with the magnitude gap between a host and its brightest satellite.** In the figure below, we now compute the total number of satellites within the projected virial radius, and plot it against the magnitude gap and host stellar mass.

<img align="center" width=480px src="assets/magnitude-gap.png" alt="The satellite abudance within the virial radius as a function of magnitude gap and host stellar mass. Figure 12 from the xSAGA I paper."/>  

**Our findings agree with the SAGA Survey's results.** We compare the radial profiles of satellites around isolated host galaxies in the same stellar mass range as SAGA (10<sup>10</sup>–10<sup>11</sup> M<sub>⊙</sub>; see figure below). Our results are in excellent agreement.

<img align="center" width=480px src="assets/saga-comparison.png" alt=" Part of Figure 6 from the xSAGA I paper."/>  


### Figures

All figures presented in our paper can be reproduced by running the cells in the `paper-figures.ipynb` Jupyter notebook. The notebook can be accessed through the [nbviewer website](https://nbviewer.org/github/jwuphysics/xSAGA/blob/main/paper-figures.ipynb).

# Code

# Citation

