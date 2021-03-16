#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from os.path import join as path_join

from config import use_data, start_year, final_year, latitude , longitude, \
                   DOWA_data_dir, location, \
                   era5_data_dir, model_level_file_name_format, surface_file_name_format, read_model_level_up_to, era5_grid_size, height_range

from era5_ml_height_calc import compute_level_heights

import dask
# only as many threads as requested CPUs | only one to be requested, more threads don't seem to be used
dask.config.set(scheduler='synchronous')


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
    ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values  # Model level numbers.
    hours = ds['time'].values

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive. Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level

def get_wind_data_era5(heights_of_interest, lat=40, lon=1, start_year=2010, final_year=2010, max_level=112):
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
        'wind_speed_east': v_req_alt_east,
        'wind_speed_north': v_req_alt_north,
        'n_samples': len(hours),
        'datetime': ds['time'].values,
        'altitude': heights_of_interest,
        'years': (start_year, final_year)
    }

    return wind_data


def get_wind_data():
    # Get actual lat/lon from chosen DOWA indices - change this in the future to the other way around? TODO
    from read_data.dowa import lats_dowa_grid, lons_dowa_grid
    lat = lats_dowa_grid[location['i_lat'], location['i_lon']]
    lon = lons_dowa_grid[location['i_lat'], location['i_lon']]
    # round to grid spacing in ERA5 data
    latitude = round(lat/era5_grid_size)*era5_grid_size
    longitude = round(lon/era5_grid_size)*era5_grid_size

    data_info = '_lat_{:2.2f}_lon_{:2.2f}'.format(latitude,longitude)

    if use_data == 'DOWA':
        data_info = '_lat_{:2.2f}_lon_{:2.2f}'.format(lat,lon) # part of FIX 

        import os
        #HDF5 library has been updated (1.10.1) (netcdf uses HDF5 under the hood)
        #file system does not support the file locking that the HDF5 library uses.
        #In order to read your hdf5 or netcdf files, you need set this environment variable :
        os.environ["HDF5_USE_FILE_LOCKING"]="FALSE" # check - is this needed? if yes - where set, needed for era5? FIX
        from read_data.dowa import read_data
        wind_data = read_data(location, DOWA_data_dir)

        # Use start_year to final_year data only
        hours = wind_data['datetime']
        start_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(start_year))
        end_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(final_year+1))

        start_idx = list(hours).index(start_date)
        end_idx = list(hours).index(end_date)
        data_range = range(start_idx, end_idx + 1)

        for key in ['wind_speed_east', 'wind_speed_north', 'datetime']:
            wind_data[key] = wind_data[key][data_range]
        wind_data['n_samples'] = len(data_range)
        wind_data['years'] = (start_year, final_year)

        data_info += "_DOWA_{}_{}".format(start_year, final_year)

        print(len(hours))
        print(len(wind_data['wind_speed_east']), wind_data['n_samples'])
    elif use_data == 'LIDAR':
        from read_data.fgw_lidar import read_data
        wind_data = read_data()
        data_info += "_LIDAR"

    elif use_data == 'ERA5':
        wind_data = get_wind_data_era5(height_range, lat=latitude, lon=longitude, start_year=start_year, final_year=final_year, max_level=read_model_level_up_to)
        data_info += "_era5_{}_{}_grid_{}".format(start_year, final_year, era5_grid_size)
    else:
        raise ValueError("Wrong data type specified: {} - no option to read data is executed".format(use_data))

    return wind_data, data_info

