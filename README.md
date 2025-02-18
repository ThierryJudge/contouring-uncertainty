# Contour Uncertainty
Echocardiography contour uncertainty estimation. 

The respository contains the code for Asymmetric Contour Uncertainty Estimation for Medical Image Segmentation (MICCAI 2023) and 
Uncertainty Propagation for Echocardiography Clinical Metric Estimation via Contour Sampling. 


This project also uses code from the [vital submodule](https://github.com/nathanpainchaud/vital).

## How to run
First, install dependencies
```bash
git clone  https://github.com/ThierryJudge/contouring-uncertainty.git

# install echo-segmentation-uncertainty
cd contour-uncertainty
conda env create -f requirements/environment.yml
 ```
You are now ready to import modules and run code.


## Scripts 

### Runner

The runner.py scripts handles training and evaluation. [Hydra](https://hydra.cc/) is used to handle the different configurations. 

To see the different configurations, use 

```bash
python runner.py -h 
 ```

## MICCAI 2023


## TMI 2025
