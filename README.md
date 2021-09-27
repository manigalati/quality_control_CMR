# Efficient model monitoring for quality control in cardiac image segmentation

Repository for code from the paper "Efficient model monitoring for quality control in cardiac image segmentation". 

It contains all the scripts and utility programs for implementing our framework.

Authors: Francesco Galati and Maria A. Zuluaga.

## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)
 * nibabel 
 * medpy

## Running the Software

All the python classes and functions strictly needed to implement the framework can be found in `CA.py`.

We demonstrate its effectiveness by reproducing the results of the [ACDC Challenge] in `QC.ipynb`, in the absence of ground truth (check also `QC_wDataAugmentation.ipynb` for a more robust implementation with the use of data augmentation techniques).

## How to cite

If you use this software, please cite the following paper as appropriate:

    Galati, F., Zuluaga, M. A. (2021).
    Efficient model monitoring for quality control in cardiac image segmentation.
    In: 11th Biennal Conference on Functional Imaging and Modeling of the Heart (FIMH)
    https://arxiv.org/abs/2104.05533

## Copyright and licensing

Copyright 2020.

This software is released under the BSD-3 license. Please see the license file_ for details.

[ACDC Challenge]: https://www.creatis.insa-lyon.fr/Challenge/acdc
