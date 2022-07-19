#!/usr/bin/env anaconda2
# -*- coding: utf-8 -*-


import numpy as np
from numpy import sinc, pi, sin, cos, arctan2, empty, ma, ceil, errstate, ones, arange, meshgrid
from scipy.special import j1
from cv2 import filter2D
from netCDF4 import Dataset
from datetime import datetime, timedelta

def distance(lon0, lat0, lon1, lat1):
    """
    Compute distance between points from each line.
    :param float lon0:
    :param float lat0:
    :param float lon1:
    :param float lat1:
    :return: distance (in m)
    :rtype: array
    """
    D2R = pi / 180.0
    sin_dlat = sin((lat1 - lat0) * 0.5 * D2R)
    sin_dlon = sin((lon1 - lon0) * 0.5 * D2R)
    cos_lat1 = cos(lat0 * D2R)
    cos_lat2 = cos(lat1 * D2R)
    a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
    return 6370997.0 * 2 * arctan2(a_val ** 0.5, (1 - a_val) ** 0.5)

def get_step_in_km(lat, wave_length, xstep, ystep):
    
    step_y_km = ystep * distance(0, 0, 0, 1) / 1000
    step_x_km = xstep * distance(0, lat, 1, lat) / 1000
    min_wave_length = max(step_x_km, step_y_km) * 2
    if wave_length < min_wave_length:
        raise Exception()
    return step_x_km, step_y_km

def estimate_kernel_shape(lat, wave_length, order, x_c, y_c):
        xstep = (x_c[1:] - x_c[:-1]).mean()
        ystep = (y_c[1:] - y_c[:-1]).mean()
        step_x_km, step_y_km = get_step_in_km(lat, wave_length, xstep, ystep)
        # half size will be multiply with by order
        half_x_pt, half_y_pt = (
            ceil(wave_length / step_x_km).astype(int),
            ceil(wave_length / step_y_km).astype(int),
        )
        # x size is not good over 60 degrees
        y = arange(
            lat - ystep * half_y_pt * order,
            lat + ystep * half_y_pt * order + 0.01 * ystep,
            ystep,
        )
        # We compute half + 1 and the other part will be compute by symetry
        x = arange(0, xstep * half_x_pt * order + 0.01 * xstep, xstep)
        y, x = meshgrid(y, x)
        dist_norm = distance(0, lat, x, y) / 1000.0 / wave_length
        return half_x_pt, half_y_pt, dist_norm
    
def finalize_kernel(kernel, order, half_x_pt, half_y_pt):
        # Symetry
        kernel_ = empty((half_x_pt * 2 * order + 1, half_y_pt * 2 * order + 1))
        kernel_[half_x_pt * order :] = kernel
        kernel_[: half_x_pt * order] = kernel[:0:-1]
        # remove unused row/column
        k_valid = kernel_ != 0
        x_valid = np.where(k_valid.sum(axis=1))[0]
        x_slice = slice(x_valid[0], x_valid[-1] + 1)
        y_valid = np.where(k_valid.sum(axis=0))[0]
        y_slice = slice(y_valid[0], y_valid[-1] + 1)
        return kernel_[x_slice, y_slice]
    
def kernel_bessel(lat, wave_length, x_c, y_c, order=1):
        """wave_length in km
        order must be int
        """
        half_x_pt, half_y_pt, dist_norm = estimate_kernel_shape(
            lat, wave_length, order, x_c, y_c
        )
        with errstate(invalid="ignore"):
            kernel = sinc(dist_norm / order) * j1(2 * pi * dist_norm) / dist_norm
        kernel[0, half_y_pt * order] = pi
        kernel[dist_norm > order] = 0
        return finalize_kernel(kernel, order, half_x_pt, half_y_pt)

def convolve_filter_with_dynamic_kernel(
    grid, kernel_func, x_c, y_c, lat_max=85, extend=False, **kwargs_func):
    """
    :param str grid: grid name
    :param func kernel_func: function of kernel to use
    :param float lat_max: absolute latitude above no filtering apply
    :param bool extend: if False, only non masked value will return a filtered value
    :param dict kwargs_func: look at kernel_func
    :return: filtered value
    :rtype: array
    """


    # Matrix for result
    data = grid.copy()
    data_out = ma.empty(data.shape)
    data_out.mask = np.ones(data_out.shape, dtype=bool)
    nb_lines = y_c.shape[0]
    dt = list()

    debug_active = False

    for i, lat in enumerate(y_c):
        if abs(lat) > lat_max or data[:, i].mask.all():
            data_out.mask[:, i] = True
            continue
        # Get kernel
        kernel = kernel_func(lat=lat, x_c=x_c, y_c=y_c, **kwargs_func)
        # Kernel shape
        k_shape = kernel.shape
        t0 = datetime.now()
        if debug_active and len(dt) > 0:
            dt_mean = np_mean(dt) * (nb_lines - i)
            print(
                "Remain ",
                dt_mean,
                "ETA ",
                t0 + dt_mean,
                "current kernel size :",
                k_shape,
                "Step : %d/%d    " % (i, nb_lines),
                end="\r",
            )

        # Half size, k_shape must be always impair
        d_lat = int((k_shape[1] - 1) / 2)
        d_lon = int((k_shape[0] - 1) / 2)
        # Temporary matrix to have exact shape at outuput
        tmp_matrix = ma.zeros((2 * d_lon + data.shape[0], k_shape[1]))
        tmp_matrix.mask = ones(tmp_matrix.shape, dtype=bool)
        # Slice to apply on input data
        # +1 for upper bound, to take in acount this column
        sl_lat_data = slice(max(0, i - d_lat), min(i + d_lat + 1, data.shape[1]))
        # slice to apply on temporary matrix to store input data
        sl_lat_in = slice(
            d_lat - (i - sl_lat_data.start), d_lat + (sl_lat_data.stop - i)
        )
        # Copy data
        tmp_matrix[d_lon:-d_lon, sl_lat_in] = data[:, sl_lat_data]
        # Convolution
        m = ~tmp_matrix.mask
        tmp_matrix[~m] = 0

        demi_x, demi_y = k_shape[0] // 2, k_shape[1] // 2
        values_sum = filter2D(tmp_matrix.data, -1, kernel)[demi_x:-demi_x, demi_y]
        kernel_sum = filter2D(m.astype(float), -1, kernel)[demi_x:-demi_x, demi_y]
        with errstate(invalid="ignore", divide="ignore"):
            if extend:
                data_out[:, i] = ma.array(
                    values_sum / kernel_sum,
                    mask=kernel_sum < (extend * kernel.sum()),
                )
            else:
                data_out[:, i] = values_sum / kernel_sum
        dt.append(datetime.now() - t0)
        if len(dt) == 100:
            dt.pop(0)
    if extend:
        out = ma.array(data_out, mask=data_out.mask)
    else:
        out = ma.array(data_out, mask=data.mask + data_out.mask)
    if debug_active:
        print()
    if out.dtype != data.dtype:
        return out.astype(data.dtype)
    return out

def bessel_high_filter(data, wave_length, y_c, x_c, order=1, lat_max=85, **kwargs):
        """
        :param str grid_name: grid to filter, data will replace original one
        :param float wave_length: in km
        :param int order: order to use, if > 1 negative values of the cardinal sinus are present in kernel
        :param float lat_max: absolute latitude, no filtering above
        :param dict kwargs: look at :py:meth:`RegularGridDataset.convolve_filter_with_dynamic_kernel`

        .. minigallery:: py_eddy_tracker.RegularGridDataset.bessel_high_filter
        """
        data_out = convolve_filter_with_dynamic_kernel(
            data,
            kernel_bessel,
            lat_max=lat_max,
            wave_length=wave_length,
            order=order,
            x_c=x_c,
            y_c=y_c, 
            **kwargs,
        )
        return data - data_out