# Traffic Classification directly on spectrum data
This repository contains the code the can be used to reproduce the results in the paper: "A General Approach for Traffic Classification in Wireless Networks using Deep Learning".  
- The dataset can be downladed from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5208201.svg)](https://doi.org/10.5281/zenodo.5208201)
- The preprint (paper still under review) can be found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5236573.svg)](https://doi.org/10.5281/zenodo.5236573)
- The last release of this code can be found at [![DOI](https://zenodo.org/badge/396962821.svg)](https://zenodo.org/badge/latestdoi/396962821)
- The published version of the paper can be found at [https://doi.org/10.1109/TNSM.2021.3130382](https://doi.org/10.1109/TNSM.2021.3130382)

This repo will be updated according to the following planning:  

## Python code  
- ~~Notebook with code showing how to load and partition the dataset for the training and testing the models~~ -> Done
- ~~Notebook with code showing how to train the CNN and RNN~~ -> Done
- ~~Notebook with code showing how to test pretrained models and to get the metrics/results~~ -> Done
- Notebook with code showing how to train a CNN and Gradient Boost based algorithm on the byte representation of the data.

## Matlab code
-Scripts to generate waveforms using L2 (bytes) payload.  


## Citing us
If you use (part of) the provided dataset or code, please dont forget to cite us as follows:  

M. Camelo, P. Soto and S. Latré, "A General Approach for Traffic Classification in Wireless Networks using Deep Learning," in IEEE Transactions on Network and Service Management, doi: 10.1109/TNSM.2021.3130382.

@ARTICLE{cameloTCW21,
  author={Camelo, Miguel and Soto, Paola and Latré, Steven},
  journal={IEEE Transactions on Network and Service Management}, 
  title={A General Approach for Traffic Classification in Wireless Networks using Deep Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSM.2021.3130382}}
