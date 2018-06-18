from collections import namedtuple
import datetime
import io
import logging
import math
import os
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyproj
from pytz import timezone, utc
from progress.bar import Bar
import scipy.optimize
from scipy.ndimage.filters import convolve1d

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.minidoas import MiniDoasException
from spectroscopy.datamodel import (PreferredFluxBuffer,
                                    InstrumentBuffer,
                                    TargetBuffer,
                                    MethodBuffer)
from spectroscopy.util import vec2bearing


logging.basicConfig(filename='minidoas2datamodel.log', filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')


GaussianParameters = namedtuple('GaussianParameters',
                                ['amplitude', 'mean', 'sigma'])


def gaussian_pts(xpts, p):
    """
    Returns the values of a Gaussian (described by the GaussianParameters p)
    corresponding to the x values passed in as xpts (an array).
    """
    xpts = np.array(xpts)
    f = gaussian_func(p)
    return f(xpts)


def gaussian_func(p):
    return lambda x: p[0]*scipy.exp(-(x-p[1])**2/(2.0*p[2]**2))


class FittingError(RuntimeError):
    pass


def __errfunc(p, x, y):
    # Ddefine an error function such that there is a high cost to having a
    # negative amplitude
    diff = (p[0]*scipy.exp(-(x-p[1])**2/(2.0*p[2]**2))) - y
    if p[0] < 0.0:
        diff *= 10000.0

    return diff


# function taken from avoscan.processing module
def fit_gaussian(xdata, ydata, amplitude_guess=None, mean_guess=None,
                 sigma_guess=None):
    """
    Fits a gaussian to some data using a least squares fit method.
    Returns a named tuple of best fit parameters (amplitude, mean, sigma).

    Initial guess values for the fit parameters can be specified as kwargs.
    Otherwise they are estimated from the data.
    """

    if len(xdata) != len(ydata):
        raise ValueError("Lengths of xdata and ydata must match")

    if len(xdata) < 4:
        msg = "xdata and ydata need to contain at least 4 elements each"
        raise ValueError(msg)

    # guess some fit parameters - unless they were specified as kwargs
    if amplitude_guess is None:
        amplitude_guess = max(ydata)

    if mean_guess is None:
        weights = ydata - np.average(ydata)
        weights[np.where(weights < 0)] = 0
        mean_guess = np.average(xdata, weights=weights)

    # Find width at half height as estimate of sigma
    if sigma_guess is None:
        # Fast and numerically precise
        variance = np.dot(np.abs(ydata),
                          (xdata-mean_guess)**2)/np.abs(ydata).sum()
        sigma_guess = math.sqrt(variance)

    # Put guess params into an array ready for fitting
    p0 = np.array([amplitude_guess, mean_guess, sigma_guess])

    # do the fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1, success = scipy.optimize.leastsq(__errfunc, p0,
                                             args=(xdata, ydata))

    if success not in (1, 2, 3, 4):
        raise FittingError("Could not fit Gaussian to data.")

    return GaussianParameters(*p1)


def binomial_filter(y, mask, order):
    y_old = y.copy()
    mask = np.array(mask)
    factor = mask.sum()
    for i in range(order):
        y_new = convolve1d(y_old, mask, mode='reflect')/factor
        y_old = y_new
    return y_new


class MDOASException(Exception):
    pass


def read_single_station(d, station_info, date):
    """
    Read all the data for a single MiniDoas station for one day.
    """
    nztz = timezone('Pacific/Auckland')
    date_nz = nztz.localize(datetime.datetime(date.year, date.month,
                                              date.day, 6, 0, 0))
    timeshift = int(date_nz.utcoffset().seconds/3600.)
    datestr = '{:d}-{:02d}-{:02d}'.format(date.year, date.month, date.day)

    # Read the raw data
    if station_info['files']['raw'] is None:
        # There is no point continuing if we don't have any raw data
        msg = "INFO 01: No raw data for:\n"
        msg += "-->Station: {}\n".format(station_info['stationID'])
        msg += "-->Date: {}\n".format(str(date))
        logging.info(msg)
        return

    e0 = d.read(station_info['files']['raw'],
                ftype='minidoas-raw', timeshift=timeshift)
    ib = InstrumentBuffer(name=station_info['stationID'],
                          location=station_info['stationLoc'],
                          no_bits=16,
                          type='MiniDOAS')
    i = d.new(ib)
    try:
        rdt = d.elements['RawDataType'][0]
    except:
        rdt = d.new(e0['RawDataTypeBuffer'])
    rb = e0['RawDataBuffer']
    rb.type = rdt
    rb.instrument = i
    rb.target = station_info['target']
    lat = np.ones(rb.d_var.shape[0])*station_info['lat']
    lon = np.ones(rb.d_var.shape[0])*station_info['lon']
    elev = np.ones(rb.d_var.shape[0])*station_info['elev']
    bearing = np.ones(rb.d_var.shape[0])*np.rad2deg(station_info['bearing'])
    rb.position = np.array([lon, lat, elev]).T
    rb.bearing = bearing
    rb.inc_angle_error = np.ones(rb.d_var.shape[0])*0.013127537*180./np.pi
    rr = d.new(rb)

    # Read the concentration
    if station_info['files']['spectra'] is None:
        msg = "INFO 02: No concentration (i.e. spectra) data for:\n"
        msg += "-->Station: {}\n".format(station_info['stationID'])
        msg += "-->Date: {}\n".format(str(date))
        logging.info(msg)
        return

    # First read in the smoothed version of the concentration
    # which the subsequent computation of flux values is
    # based on
    e1 = d.read(station_info['files']['spectra'],
                date=datestr, ftype='minidoas-spectra',
                timeshift=timeshift, model=True)
    cb = e1['ConcentrationBuffer']
    idxs = np.zeros(cb.value.shape)
    for i in range(cb.value.shape[0]):
        idx = np.argmin(np.abs(rr.datetime[:].astype('datetime64[ms]')
                               - cb.datetime[i].astype('datetime64[ms]')))
        idxs[i] = idx
    cb.rawdata = [rr]
    cb.rawdata_indices = idxs
    cb.method = station_info['widpro_method']
    cb.user_notes = 'smoothed path concentration'
    cc = d.new(cb)

    # Now read in the original path concentration
    # to keep as a reference
    e2 = d.read(station_info['files']['spectra'],
                date=datestr, ftype='minidoas-spectra',
                timeshift=timeshift)
    cb2 = e2['ConcentrationBuffer']
    idxs = np.zeros(cb2.value.shape)
    for i in range(cb.value.shape[0]):
        idx = np.argmin(np.abs(rr.datetime[:].astype('datetime64[ms]')
                               - cb2.datetime[i].astype('datetime64[ms]')))
        idxs[i] = idx
    cb2.rawdata = [rr]
    cb2.rawdata_indices = idxs
    cb2.method = station_info['widpro_method']
    cb2.user_notes = 'original path concentration'

    # Read in the flux estimates for assumed height
    if station_info['files']['flux_ah'] is None:
        msg = "INFO 03: No assumed height flux data for:\n"
        msg += "-->Station: {}\n".format(station_info['stationID'])
        msg += "-->Date: {}\n".format(str(date))
        logging.info(msg)
    else:
        e3 = d.read(station_info['files']['flux_ah'],
                    date=datestr, ftype='minidoas-scan',
                    timeshift=timeshift)
        fb = e3['FluxBuffer']
        dt = fb.datetime[:].astype('datetime64[s]')
        indices = []
        for _dt in dt:
            idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]')
                                   - _dt))
            idx0 = idx
            while True:
                angle = rr.inc_angle[cc.rawdata_indices[idx]+1]
                if angle > 180.:
                    break
                idx += 1
            idx1 = idx
            indices.append([idx0, idx1+1])
        fb.concentration = cc
        fb.concentration_indices = indices

        gfb1 = e3['GasFlowBuffer']

        m2 = None
        for _m in d.elements['Method']:
            if _m.name[:] == 'WS2PV':
                m2 = _m
        if m2 is None:
            mb2 = e3['MethodBuffer']
            m2 = d.new(mb2)

        gfb1.methods = [m2]
        gf1 = d.new(gfb1)
        fb.gasflow = gf1
        f = d.new(fb)
        # Now read in preferred flux values for assumed
        # height downloaded from FITS
        if station_info['files']['fits_flux_ah'] is None:
            msg = "ERROR 01: No preferred flux for assumed height in FITS:\n"
            msg += "-->Station: {}\n".format(station_info['stationID'])
            msg += "-->Date: {}\n".format(str(date))
            logging.error(msg)
        else:
            data_ah = np.loadtxt(station_info['files']['fits_flux_ah'],
                                 dtype=np.dtype([('date', 'S19'),
                                                 ('val', np.float),
                                                 ('err', np.float)]),
                                 skiprows=1, delimiter=',', ndmin=1)
            dates = data_ah['date'].astype('datetime64[s]')
            indices = []
            values = []
            val_err = []
            ndates = []
            for i, dt in enumerate(dates):
                min_tdiff = np.min(np.abs(f.datetime[:].astype('datetime64[s]')
                                          - dt))
                if min_tdiff.astype('int') > 1:
                    msg = "ERROR 02: No assumed height flux estimate can be"
                    msg += " found for FITS value:\n"
                    msg += "-->Station: {}\n".format(station_info['stationID'])
                    msg += "-->Date: {}\n".format(str(dt))
                    msg += "-->FITS value: {}\n".format(data_ah['val'][i])
                    logging.error(msg)
                else:
                    idx = np.argmin(np.abs(f.datetime[:].
                                           astype('datetime64[s]') - dt))
                    indices.append(idx)
                    values.append(data_ah['val'][i])
                    val_err.append(data_ah['err'][i])
                    ndates.append(str(dt))
            if len(indices) > 0:
                pfb = PreferredFluxBuffer(fluxes=[f],
                                          flux_indices=[indices],
                                          value=values,
                                          value_error=val_err,
                                          datetime=ndates)
                d.new(pfb)

    # Read in the flux estimates for calculated height
    if station_info['files']['flux_ch'] is None:
        msg = "INFO 04: No calculated height flux data for:\n"
        msg += "-->Station: {}\n".format(station_info['stationID'])
        msg += "-->Date: {}\n".format(str(date))
        logging.info(msg)
    else:
        e4 = d.read(station_info['files']['flux_ch'],
                    date=datestr, ftype='minidoas-scan',
                    station=station_info['wp_station_id'],
                    timeshift=timeshift)
        fb1 = e4['FluxBuffer']
        dt = fb1.datetime[:].astype('datetime64[s]')
        indices = []
        for _dt in dt:
            idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]')
                                   - _dt))
            idx0 = idx
            while True:
                angle = rr.inc_angle[cc.rawdata_indices[idx]+1]
                if angle > 180.:
                    break
                idx += 1
            idx1 = idx
            indices.append([idx0, idx1])
        fb1.concentration = cc
        fb1.concentration_indices = indices

        m3 = None
        for _m in d.elements['Method']:
            if _m.name[:] == 'WS2PVT':
                m3 = _m
        if m3 is None:
            mb3 = e4['MethodBuffer']
            new_description = mb3.description[0]
            new_description += '; plume geometry inferred from triangulation'
            mb3.description = new_description
            mb3.name = 'WS2PVT'
            m3 = d.new(mb3)

        gfb2 = e4['GasFlowBuffer']
        gfb2.methods = [m3]
        gf2 = d.new(gfb2)
        fb1.gasflow = gf2
        f1 = d.new(fb1)

        # Now read in preferred flux values for calculated
        # height downloaded from FITS
        if station_info['files']['fits_flux_ch'] is None:
            msg = "ERROR 01: No preferred flux for"
            msg = " calculated height in FITS:\n"
            msg += "-->Station: {}\n".format(station_info['stationID'])
            msg += "-->Date: {}\n".format(str(date))
            logging.error(msg)
        else:
            data_ch = np.loadtxt(station_info['files']['fits_flux_ch'],
                                 dtype=np.dtype([('date', 'S19'),
                                                 ('val', np.float),
                                                 ('err', np.float)]),
                                 skiprows=1, delimiter=',', ndmin=1)
            dates = data_ch['date'].astype('datetime64[s]')
            indices = []
            values = []
            val_err = []
            ndates = []
            for i, dt in enumerate(dates):
                min_tdiff = np.min(np.abs(f1.datetime[:].
                                          astype('datetime64[s]') - dt))
                if min_tdiff.astype('int') > 1:
                    msg = "ERROR 02: No calculated height flux estimate can be"
                    msg = " found for FITS value:\n"
                    msg += "-->Station: {}\n".format(station_info['stationID'])
                    msg += "-->Date: {}\n".format(str(dt))
                    msg += "-->FITS value: {}\n".format(data_ah['val'][i])
                    logging.error(msg)
                else:
                    idx = np.argmin(np.abs(f1.datetime[:].
                                           astype('datetime64[s]') - dt))
                    indices.append(idx)
                    values.append(data_ch['val'][i])
                    val_err.append(data_ch['err'][i])
                    ndates.append(str(dt))
            if len(indices) > 0:
                pfb1 = PreferredFluxBuffer(fluxes=[f1],
                                           flux_indices=[indices],
                                           value=values,
                                           value_error=val_err,
                                           datetime=ndates)
                d.new(pfb1)


def flux_ah(pf, perror_thresh=0.5, smoothing=False):
    """
    Compute the flux for assumed height.
    """

    f = pf.fluxes[0]
    c = f.concentration
    errors = []
    for i, fidx in enumerate(pf.flux_indices[0][:]):
        idx0, idx1 = f.concentration_indices[fidx]
        so2_raw = c.value[idx0:idx1]
        r = c.rawdata
        ridx = c.rawdata_indices[idx0:idx1]
        angles = r[0].inc_angle[ridx]
        bearing = r[0].bearing[0]

        if smoothing:
            so2_smooth = binomial_filter(so2_raw, [1, 2, 1], 5)
            p = fit_gaussian(angles, so2_smooth)
            so2 = gaussian_pts(angles, p)
        else:
            so2 = so2_raw

        # Get the wind direction
        gf = f.gasflow
        gf_times = f.gasflow.datetime[:].astype('datetime64[s]')
        gf_idx = np.argmin(np.abs(gf_times - np.datetime64(f.datetime[fidx])))
        vx, vy = gf.vx[gf_idx], gf.vy[gf_idx]
        wd = vec2bearing(vx, vy)
        ws = np.sqrt(vx*vx + vy*vy)

        t_angle = np.cos(np.deg2rad(bearing - wd - 90.))

        _, _, elev0 = gf.position[gf_idx]
        _, _, elev2 = gf.position[gf_idx+2]
        plumewidth = elev2-elev0

        plmax = np.argmax(so2)
        edge = 0.05*so2.max()
        pidx = np.where(so2 >= edge)
        plstart = pidx[0][0]
        plend = pidx[0][-1]
        plrange = np.abs(np.deg2rad(angles[plstart])
                         - np.deg2rad(angles[plend]))
        # calculate distance between measurements assuming dx = r * theta
        int_time = r[0].integration_time[ridx]/1000.
        dx = (np.ones(angles.size) * 0.015707963 * 1000./992 *
              int_time*plumewidth/plrange)

        col_amt = dx[plstart:plend+1] * so2[plstart:plend+1]
        ica = np.sum(col_amt)

        # correct for a non-perpendicular transect through the plume
        ica = abs(t_angle * ica)
        flux = ica * ws * 0.000230688
        # Compute percentage error
        x = pf.value[i]*86.4
        x0 = flux
        p_error = 100.*abs(x0-x)/x
        if p_error > perror_thresh:
            msg = "ERROR 03: Error of {:.3f} exceeds threshold"
            msg += " for assumed height flux.\n"
            msg += "Date: {:s}\n"
            msg += "Expected flux (FITS): {:.3f}; Original flux: {:.3f}; "
            msg += "Estimated flux: {:.3f}\n"
            msg += "Plume geometry: start={:.3f}, max={:.3f}, end={:.3f}"
            msg += ", width={:.3f}\n"
            msg += "Wind: track={:.3f} sp={:.3f}\n"
            msg = msg.format(p_error, f.datetime[fidx], x,
                             f.value[fidx]*86.4, x0,
                             np.deg2rad(angles[plstart]),
                             np.deg2rad(angles[plmax]),
                             np.deg2rad(angles[plend]),
                             plumewidth, wd, ws)
            errors.append(msg)
    if len(errors) > 0:
        raise MDOASException(''.join(errors))


def flux_ch(pf_ch, pf_ah, perror_thresh=0.5, smoothing=False):
    """
    Compute the flux for calculated (i.e. triangulated) height.
    """
    f_ch = pf_ch.fluxes[0]
    c_ch = f_ch.concentration
    r_ch = f_ch.concentration.rawdata[0]
    gf_ch = f_ch.gasflow

    g = pyproj.Geod(ellps='WGS84')

    gf_ah = pf_ah.fluxes[0].gasflow
    f_ah = pf_ah.fluxes[0]
    c_ah = c_ch
    r_ah = r_ch

    for i, fidx in enumerate(pf_ch.flux_indices[0][:]):
        dt = f_ch.datetime[fidx].astype('datetime64[s]')
        min_tdiff = np.min(np.abs(f_ah.datetime[:].astype('datetime64[s]')
                                  - dt)).astype('int')
        # Set an arbitrary maximum value of 1 s difference
        # between the assumed height and calculated height difference
        # This is supposed to catch cases where we have a calculated height
        # estimate without an assumed height estimate
        if min_tdiff > 1.:
            msg = "ERROR 04: No assumed height estimate found for"
            msg += " calculated height estimate.\n"
            msg += "Date: {}\n".format(str(dt))
            msg += "Time difference to assumed"
            msg += " height estimate: {} s\n".format(min_tdiff)
            raise MDOASException(msg)

        idx = np.argmin(np.abs(f_ah.datetime[:].astype('datetime64[s]') - dt))
        _, _, h = gf_ch.position[fidx*3+1]
        lon, lat, h1 = gf_ah.position[idx*3+1]
        idx0, idx1 = f_ah.concentration_indices[idx]

        so2_raw = c_ah.value[idx0:idx1]
        ridx = c_ah.rawdata_indices[idx0:idx1]
        angles = r_ah.inc_angle[ridx]

        if smoothing:
            so2_smooth = binomial_filter(so2_raw, [1, 2, 1], 5)
            p = fit_gaussian(angles, so2_smooth)
            so2 = gaussian_pts(angles, p)
        else:
            so2 = so2_raw

        idx_max = np.argmax(so2)
        ridx_max = c_ah.rawdata_indices[idx0+idx_max]
        angle_max = r_ah.inc_angle[ridx_max]
        lon1, lat1, h2 = r_ah.position[ridx_max]
        _, _, dist = g.inv(lon, lat, lon1, lat1)
        pr_old = np.sqrt(dist*dist)
        pr = h/np.sin(np.deg2rad(angle_max))
        rangescale = pr/pr_old
        # Compute percentage error
        x = pf_ch.value[i]*86.4
        x0 = f_ah.value[idx]*rangescale*rangescale*86.4
        p_error = 100.*abs(x0-x)/x
        if p_error > perror_thresh:
            msg = "ERROR 03: Error of {} exceeds".format(p_error)
            msg += " threshold for calculated height flux.\n"
            msg += "Date: {}\n".format(str(dt))
            msg += "Time difference to assumed"
            msg += " height estimate: {} s\n".format(min_tdiff)
            msg += "Expected flux: {}; Estimated flux: {}.\n".format(x, x0)
            raise MDOASException(msg)


def verify_flux(filename, perror_thresh=0.5):
    """
    Verify that the recomputed flux values are close to the
    flux values stored in FITS.
    """
    d = Dataset.open(filename)
    pf_ah = None
    pf_ch = None
    for instrument in ['WI301', 'WI302']:
        for _pf in d.elements['PreferredFlux']:
            gfm = _pf.fluxes[0].gasflow.methods[0].name[:]
            sid = _pf.fluxes[0].concentration.rawdata[0].instrument.name[:]
            if gfm == 'WS2PV' and sid == instrument:
                pf_ah = _pf
            elif gfm == 'WS2PVT' and sid == instrument:
                pf_ch = _pf
        try:
            if pf_ah is not None:
                flux_ah(pf_ah, perror_thresh=perror_thresh)
                if pf_ch is not None:
                    flux_ch(pf_ch, pf_ah, perror_thresh=perror_thresh)
        except MDOASException as e:
            msg = str(e)
            msg += "StationID: {}\n".format(instrument)
            msg += "File: {}\n".format(filename)
            raise MDOASException(msg)
    d.close()


def is_file_OK(filename):
    """
    Check that file exist and contains more than
    just a header line.
    """
    if filename is None:
        return False
    if os.path.isfile(filename):
        with io.open(filename, encoding='utf-8-sig') as fh:
            linecount = len(fh.readlines())
        if linecount < 2:
            return False
    else:
        logging.error("File {} does not exist".format(filename))
        return False
    return True


def FITS_download(date, station, outputpath='/tmp'):
    st_lookup = {'WI301': 'NE', 'WI302': 'SR'}
    nztz = timezone('Pacific/Auckland')
    date_start = nztz.localize(datetime.datetime(date.year,
                                                 date.month,
                                                 date.day, 0, 0, 0))
    date_end = nztz.localize(datetime.datetime(date.year,
                                               date.month,
                                               date.day, 23, 59, 59))
    date_start_utc = date_start.astimezone(utc)
    date_end_utc = date_end.astimezone(utc)
    base_url = "https://fits.geonet.org.nz/observation"
    url = "{}?siteID={}&typeID=SO2-flux-a&methodID={}"
    filepaths = []
    for method in ['mdoas-ah', 'mdoas-ch']:
        _type = method.split('-')[1]
        filename = 'FITS_{}_{:d}{:02d}{:02d}_{}.csv'.format(st_lookup[station],
                                                            date.year,
                                                            date.month,
                                                            date.day,
                                                            _type)
        filepath = os.path.join(outputpath, filename)
        fits_cache = os.path.join('/tmp', "{}_{}.csv".format(station, method))
        if os.path.isfile(fits_cache):
            df = pd.read_csv(fits_cache, index_col=0, parse_dates=True,
                             skiprows=1, names=['obs', 'error'])
        else:
            df = pd.read_csv(url.format(base_url, station, method),
                             index_col=0, parse_dates=True,
                             skiprows=1, names=['obs', 'error'])
            df.to_csv(fits_cache)
        df_new = df.loc[(df.index >= date_start_utc) &
                        (df.index <= date_end_utc)]
        if df_new.size < 1:
            filepath = None
        else:
            df_new.to_csv(filepath)
        filepaths.append(filepath)
    return filepaths


def main(datapath, outputpath, start, end, pg=True, deletefiles=False):
    msg = "Data path is: {}\n".format(datapath)
    msg += "Output path is: {}\n".format(outputpath)
    msg += "Start date: {}\n".format(start)
    msg += "End date: {}\n".format(end)
    logging.info(msg)
    dates = pd.date_range(start=start, end=end, freq='D')
    if pg:
        ndays = len(dates)
        bar = Bar('Processing', max=ndays)

    for date in dates:
        if pg:
            bar.next()
        else:
            print(date)
        outputfile = 'MiniDOAS_{:d}{:02d}{:02d}.h5'.format(date.year,
                                                           date.month,
                                                           date.day)
        h5file = os.path.join(outputpath, outputfile)
        if True:
            d = Dataset(h5file, 'w')

            # ToDo: get correct plume coordinates
            tb = TargetBuffer(name='White Island main plume',
                              target_id='WI001',
                              position=[177.18375770, -37.52170799, 321.0])
            t = d.new(tb)

            wpoptions = "{'Pixel316nm':479, 'TrimLower':30, 'LPFilterCount':3,"
            wpoptions += "'MinWindSpeed':3,'BrightEnough':500, 'BlueStep':5, "
            wpoptions += "'MinR2:0.8, 'MaxFitCoeffError':50.0, "
            wpoptions += "'InPlumeThresh':0.05, 'MinPlumeAngle':0.1, "
            wpoptions += "'MaxPlumeAngle':3.0, 'MinPlumeSect':0.4, "
            wpoptions += "'MaxPlumeSect':2.0, 'MeanPlumeCtrHeight':310, "
            wpoptions += "'SEMeanPlumeCtrHeight':0.442, "
            wpoptions += " 'MaxRangeSeperation':5000, 'MaxRangeToPlume':5000, "
            wpoptions += " 'MaxPlumeWidth':2600'MaxPlumeCentreAltitude':2000, "
            wpoptions += "'MaxAltSeperation':1000, 'MaxTimeDiff':30,"
            wpoptions += "'MinTriLensAngle':0.1745, 'MaxTriLensAngle':2.9671,"
            wpoptions += "'SEWindSpeed':0.20, 'WindMultiplier':1.24, "
            wpoptions += "'SEWindDir':0.174}"
            mb1 = MethodBuffer(name='WidPro v1.2',
                               description='Jscript wrapper for DOASIS',
                               settings=wpoptions)
            m1 = d.new(mb1)

            station_info = {}
            location_name = 'White Island North-East Point'
            station_info['WI301'] = {'files': {},
                                     'stationID': 'WI301',
                                     'stationLoc': location_name,
                                     'target': t,
                                     'bearing': 6.0214,
                                     'lon': 177.192979384,
                                     'lat': -37.5166903535,
                                     'elev': 49.0,
                                     'widpro_method': m1,
                                     'wp_station_id': 'NE'}

            station_info['WI302'] = {'files': {},
                                     'stationID': 'WI302',
                                     'stationLoc': 'White Island South Rim',
                                     'target': t,
                                     'bearing': 3.8223,
                                     'lon': 177.189013316,
                                     'lat': -37.5265334424,
                                     'elev': 96.0,
                                     'widpro_method': m1,
                                     'wp_station_id': 'SR'}

            for station in ['WI301', 'WI302']:

                # Find the raw data
                raw_data_filename = "{:s}_{:d}{:02d}{:02d}.zip"
                station_id = station_info[station]['wp_station_id']
                raw_data_filename = raw_data_filename.format(station_id,
                                                             date.year,
                                                             date.month,
                                                             date.day)
                raw_data_filepath = os.path.join(datapath, 'spectra',
                                                 station_id,
                                                 raw_data_filename)
                if os.path.isfile(raw_data_filepath):
                    try:
                        with ZipFile(raw_data_filepath) as myzip:
                            myzip.extractall('/tmp')
                    except:
                        msg = "ERROR 05: Can't unzip file {}"
                        logging.error(msg.format(raw_data_filepath))
                        raw_data_filepath = None
                    else:
                        raw_data_filename = raw_data_filename.replace('.zip',
                                                                      '.csv')
                        raw_data_filepath = os.path.join('/tmp',
                                                         raw_data_filename)
                else:
                    logging.error("file {} does not exist"
                                  .format(raw_data_filepath))
                    continue
                try:
                    if not is_file_OK(raw_data_filepath):
                        raw_data_filepath = None
                except Exception as e:
                    print(raw_data_filepath)
                    raise(e)
                station_info[station]['files']['raw'] = raw_data_filepath

                # Find the concentration data
                monthdir = '{:d}-{:02d}'.format(date.year, date.month)
                spectra_filename = "{:s}_{:d}_{:02d}_{:02d}_Spectra.csv"
                spectra_filename = spectra_filename.format(station_id,
                                                           date.year,
                                                           date.month,
                                                           date.day)
                spectra_filepath = os.path.join(datapath, 'results',
                                                monthdir,
                                                spectra_filename)
                if not is_file_OK(spectra_filepath):
                    spectra_filepath = None

                station_info[station]['files']['spectra'] = spectra_filepath

                # Find the flux data
                flux_ah_filename = spectra_filename.replace('Spectra.csv',
                                                            'Scans.csv')
                flux_ah_filepath = os.path.join(datapath, 'results',
                                                monthdir,
                                                flux_ah_filename)
                if not is_file_OK(flux_ah_filepath):
                    flux_ah_filepath = None

                station_info[station]['files']['flux_ah'] = flux_ah_filepath

                flux_ch_filename = "XX_{:d}_{:02d}_{:02d}_Combined.csv"
                flux_ch_filename = flux_ch_filename.format(date.year,
                                                           date.month,
                                                           date.day)
                flux_ch_filepath = os.path.join(datapath, 'results',
                                                monthdir,
                                                flux_ch_filename)
                if not is_file_OK(flux_ch_filepath):
                    flux_ch_filepath = None

                station_info[station]['files']['flux_ch'] = flux_ch_filepath

                fits_flux_ah, fits_flux_ch = FITS_download(date, station)
                station_info[station]['files']['fits_flux_ah'] = fits_flux_ah
                station_info[station]['files']['fits_flux_ch'] = fits_flux_ch

                try:
                    read_single_station(d, station_info[station], date)
                except MiniDoasException as e:
                    logging.error(str(e))

            # Wind data
            windd_dir = os.path.join(datapath, 'wind', 'direction')
            winds_dir = os.path.join(datapath, 'wind', 'speed')
            sub_dir = '{:02d}-{:02d}'.format(date.year-2000, date.month)
            winds_filename = '{:d}{:02d}{:02d}_WS_00.txt'.format(date.year,
                                                                 date.month,
                                                                 date.day)
            windd_filename = winds_filename.replace('WS', 'WD')
            winds_filepath = os.path.join(winds_dir, sub_dir, winds_filename)
            windd_filepath = os.path.join(windd_dir, sub_dir, windd_filename)

            if is_file_OK(winds_filepath) and is_file_OK(windd_filepath):
                # Read in the raw wind data; this is currently
                # not needed to reproduce flux estimates so it's
                # just stored for reference
                e = d.read({'direction': windd_filepath,
                            'speed': winds_filepath},
                           ftype='minidoas-wind', timeshift=13)
                gfb = e['GasFlowBuffer']
                gf = d.new(gfb)

            d.close()
        try:
            verify_flux(os.path.join(outputpath, outputfile), 1.)
        except MDOASException as e:
            msg = str(e)
            logging.error(msg)
        if deletefiles:
            if h5file is not None and os.path.isfile(h5file):
                os.remove(h5file)
            for station in ['WI301', 'WI302']:
                files = [station_info[station]['files']['raw'],
                         station_info[station]['files']['fits_flux_ah'],
                         station_info[station]['files']['fits_flux_ch']]
                for _f in files:
                    if _f is not None and os.path.isfile(_f):
                        os.remove(_f)
    if pg:
        bar.finish()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath',
                        help='absolute path to minidoas root directory')
    parser.add_argument('outputpath',
                        help='parent directory for output')
    parser.add_argument('start',
                        help='start date as yyy-mm-dd')
    parser.add_argument('end',
                        help='end date as yyy-mm-dd')
    parser.add_argument('--pg', help='enable progress bar',
                        action='store_true')
    parser.add_argument('-d', '--delete', help='delete files after conversion',
                        action='store_true')
    args = parser.parse_args()
    main(args.datapath, args.outputpath, args.start, args.end, pg=args.pg,
         deletefiles=args.delete)
