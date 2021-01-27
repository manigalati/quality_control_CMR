# Efficient model monitoring for quality control in cardiac image segmentation

Scripts and utility programs for implementing our framework.

Authors: Francesco Galati, Maria A. Zuluaga.

## How to cite

If you use this software, please cite the following paper as appropriate:

    Galati, F., Zuluaga, M. A. (2020).
    Efficient model monitoring for quality control in cardiac image segmentation.

## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)
 * nibabel 
 * medpy

## Running the Software

All the python classes and functions strictly needed to implement the framework can be found in `CA.py`.

We demonstrate its effectiveness by reproducing the results of the [ACDC Challenge] in `Model_Monitoring_for_Cardiac_Image_Segmentation.ipynb`, in the absence of ground truth.

## Copyright and licensing

Copyright 2020.

This software is released under the BSD-3 license. Please see the license file_ for details.

[ACDC Challenge]: https://www.creatis.insa-lyon.fr/Challenge/acdc
