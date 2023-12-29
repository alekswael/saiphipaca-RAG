#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install packages from requirements.txt
pip install -r requirements.txt

# Deactivate virtual environment
deactivate
