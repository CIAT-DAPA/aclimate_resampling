# AClimate resampling module

![GitHub release (latest by date)](https://img.shields.io/github/v/release/CIAT-DAPA/aclimate_resampling) ![](https://img.shields.io/github/v/tag/CIAT-DAPA/aclimate_resampling)

This repository contains all related to resampling module for AClimate

## Features

- Generate climate scenarios base on probabilities
- Complete daily data using CHIRPS and ERA 5
- Include data for case use in **data** folder
- Include modules to configure the env in **modules** folder

## Prerequisites

- Python 3.x
- GDAL
- You need the .cdsapirc file which should be in $HOME if you are using linux or User Home if you use Windows

## Configure DEV Enviroment

You should create a env to run the code and install the requeriments. 

### Linux

Run the following commands in the prompt

````bash
sudo add-apt-repository -y ppa:ubuntugis/ppa
sudo apt-get update -q
sudo apt-get install -y libgdal-dev

pip install virtualenv
venv env
pip install -r requirements.txt

pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
````

### Windows

You have to edit the **requirements_windows.txt** file. Change the path of the wheels for 
**GDAL** and **rasterio** package for the path of the files in **modules**. It should be absolute path.

````bash
pip install virtualenv
venv env
pip install -r requirements_windows.txt
````

## Run Test

You can run the unit testing using the following command

````bash
python -m unittest discover -s .\test
````

## Install

This module can be used as a library in other Python projects. You can install this module using pip:

````bash
pip install git+https://github.com/CIAT-DAPA/aclimate_resampling
````

If you want to download a specific version of orm you can do so by indicating the version tag (@v0.0.0) at the end of the install command 

````bash
pip install git+https://github.com/CIAT-DAPA/aclimate_resampling@v0.2.0
````

## Run

You can run the module using the command aclimate_resampling followed by the required parameters:

````bash
aclimate_resampling -C ETHIOPIA -p "D:\\aclimate_resampling\\data\\" -m 1 -c 2 -y 2023
````

### Params
-C, --country: Name of the country to be processed.
-p, --path: Root path where the forecast is running (default: current directory).
-m, --prev-months: Amount of months that you want to add.
-c, --cores: Number of CPU cores to use in the calculation.
-y, --forecast-year: Year Forecast.