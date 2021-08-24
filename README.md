# Traffic Classification directly on spectrum data
This repository contains the code the can be used to reproduce the results in the paper: "A General Approach for Traffic Classification in Wireless Networks using Deep Learning".  
- The dataset can be downladed from [https://zenodo.org/record/5208201](https://zenodo.org/record/5208201)
- The preprint (paper still under review) can be found at [https://zenodo.org/record/5236573 ](https://zenodo.org/record/5236573) 
- The last release of this code can be found at [![DOI](https://zenodo.org/badge/396962821.svg)](https://zenodo.org/badge/latestdoi/396962821)

This repo will be updated according to the following planning:  

## Python code  
- ~~Notebook with code showing how to load and partition the dataset for the training and testing the models~~ -> Done
- Notebook with code showing how to train the CNN and RNN -> WIP
- Notebook with code showing how to train a CNN and Gradient Boost based algorithm on the byte representation of the data.

## Matlab code
-Scripts to generate waveforms using L2 (bytes) payload.  


## Citing us
If you use (part of) the provided dataset or code, please dont forget to cite us as follows:  

Miguel Camelo, Paola Soto, & Steven Latré. (2021). A General Approach for Traffic Classification in Wireless Networks using Deep Learning (v1.0). Zenodo. https://doi.org/10.5281/zenodo.5236573

@misc{miguel_camelo_2021_5236573,  
  author       = {Miguel Camelo and  
                  Paola Soto and  
                  Steven Latré},  
  title        = {{A General Approach for Traffic Classification in   
                   Wireless Networks using Deep Learning}},  
  month        = aug,  
  year         = 2021,  
  publisher    = {Zenodo},  
  version      = {v1.0},  
  doi          = {10.5281/zenodo.5236573},  
  url          = {https://doi.org/10.5281/zenodo.5236573}  
}  
