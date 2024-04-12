#!/bin/bash
## Steps to resolve conda pinned specifications conflict error (Comment these out if the error is not encountered)

# Update Conda
conda update conda -y

# Clear cache
conda clean --all

# Remove pinned specification
conda remove python=3.8 -y

# Install Python 3.10 using conda
conda install python=3.10 -y

# Install gxx linux
conda install gxx_linux-64 -y

# Install dependencies from requirements.txt using pip
pip install -r requirements.txt

