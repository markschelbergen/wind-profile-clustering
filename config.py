# -*- coding: utf-8 -*-
"""Configuration file for wind resource analysis.

Attributes:
    start_year (int): Pprocess the wind data starting from this year - in four-digit format.
    final_year (int): Process the wind data up to this year - in four-digit format.
    era5_data_dir (str): Directory path for reading era5 data files.
    model_level_file_name_format (str): Target name of the wind data files. Python's format() is used to fill in year
        and month at placeholders.
    surface_file_name_format (str): Target name of geopotential and surface pressure data files. Python's format() is
        used to fill in year and month at placeholders.

.FILL
.
.


.. _L137 model level definitions:
    https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

"""

# --------------------------- GENERAL
use_data_opts = ['DOWA', 'LIDAR', 'ERA5']
use_data = use_data_opts[2]


# See plots interactively - don't save plots directly as pdf to result_dir
plots_interactive = False
result_dir = "../clustering_results/" + use_data + "/"


start_year = 2010
final_year = 2010

# Single location processing
latitude = 0
longitude = 0
# FIX: get dowa indices by lat/lon - for now: dowa loc indices used for both

# --------------------------- DOWA

DOWA_data_dir = "/gpfs/share/home/s6lathim/AWE/DOWA/" 
# "/home/mark/WindData/DOWA/"  # '/media/mark/LaCie/DOWA/'

location = {'i_lat': 110, 'i_lon': 55}


# --------------------------- ERA5 
# General settings.
era5_data_dir = '/cephfs/user/s6lathim/ERA5Data-redownload/' 
model_level_file_name_format = "{:d}_europe_{:d}_130_131_132_133_135.nc"  # 'ml_{:d}_{:02d}.netcdf'
surface_file_name_format = "{:d}_europe_{:d}_152.nc" # 'sfc_{:d}_{:02d}.netcdf' 
era5_grid_size = 1. #0.25
# Processing settings
read_model_level_up_to = 112
height_range = [ 10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160., 180.,
        200., 220., 250., 300., 500., 600.]



