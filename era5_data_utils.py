#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from os.path import join as path_join

#awe era5 processing utils
from utils import compute_level_heights

import dask # clustering env no dask - FIX before running on cluster
# only as many threads as requested CPUs | only one to be requested, more threads don't seem to be used
dask.config.set(scheduler='synchronous')

era5_data_dir = '/cephfs/user/s6lathim/ERA5Data-112/'
model_level_file_name_format = "{:d}_europe_{:d}_130_131_132_133_135.nc"  # 'ml_{:d}_{:02d}.netcdf'
surface_file_name_format = "{:d}_europe_{:d}_152.nc" # 'sfc_{:d}_{:02d}.netcdf'

def read_raw_data(start_year, final_year):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        tuple of Dataset, ndarray, ndarray, ndarray, and ndarray: Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid, model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    # Construct the list of input NetCDF files
    ml_files = []
    sfc_files = []
    for y in range(start_year, final_year+1):
        for m in range(1, 13):
            ml_files.append(path_join(era5_data_dir, model_level_file_name_format.format(y, m)))
            sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
    # Load the data from the NetCDF files.
    ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=False)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values  # Model level numbers.
    hours = ds['time'].values  # Hours since 1900-01-01 00:00:0.0, see: print(nc.variables['time']).

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive. Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level

def get_wind_data_era5(lat=40, lon=1, start_year=2010, final_year=2010, max_level=112):
    heights_of_interest = [ 10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160., 180.,
        200., 220., 250., 300., 500., 600.]
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year)
    i_lat = list(lats).index(lat)
    i_lon = list(lons).index(lon)

    i_highest_level = list(levels).index(max_level)

    # Read single location wind data 
    v_levels_east = ds['u'][:, i_highest_level:, i_lat, i_lon].values
    v_levels_north = ds['v'][:, i_highest_level:, i_lat, i_lon].values

    t_levels = ds['t'][:, i_highest_level:, i_lat, i_lon].values
    q_levels = ds['q'][:, i_highest_level:, i_lat, i_lon].values

    try:
        surface_pressure = ds.variables['sp'][:, i_lat, i_lon].values
    except KeyError:
        surface_pressure = np.exp(ds.variables['lnsp'][:, i_lat, i_lon].values)

    ds.close()  # Close the input NetCDF file.

    # Calculate model level height
    level_heights, density_levels = compute_level_heights(levels, surface_pressure, t_levels, q_levels)

    # Determine wind at altitudes of interest by means of interpolating the raw wind data.
    v_req_alt_east = np.zeros((len(hours), len(heights_of_interest)))
    v_req_alt_north = np.zeros((len(hours), len(heights_of_interest)))

    for i_hr in range(len(hours)):
        if not np.all(level_heights[i_hr, 0] > heights_of_interest):
            raise ValueError("Requested height ({:.2f} m) is higher than height of highest model level."
                             .format(level_heights[i_hr, 0]))
        v_req_alt_east[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
	                               v_levels_east[i_hr, ::-1])
        v_req_alt_north[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
	                               v_levels_north[i_hr, ::-1])

    wind_data = {
        'wind_speed_east' : v_req_alt_east,
        'wind_speed_north' : v_req_alt_north,
        'n_samples' : len(hours),
        'datetime' : hours,
        'altitude' : heights_of_interest,
        'years' : (start_year, final_year)
    }

    return wind_data
    

