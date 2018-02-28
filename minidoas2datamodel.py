import datetime
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyproj
from pytz import timezone, utc

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.minidoas import MiniDoasException
from spectroscopy.visualize import plot
from spectroscopy.datamodel import (PreferredFluxBuffer,
                                    InstrumentBuffer,
                                    TargetBuffer,
                                    MethodBuffer)
from spectroscopy.util import vec2bearing, bearing2vec


class MDOASException(Exception): pass


def read_single_station(d, station_info):
    """
    Read all the data for a single MiniDoas station for one day.
    """
    # Read the raw data
    e0 = d.read(station_info['files']['raw'], 
                ftype='minidoas-raw', timeshift=13)
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
    e1 = d.read(station_info['files']['spectra'],
                date='2016-11-01', ftype='minidoas-spectra', timeshift=13)
    cb = e1['ConcentrationBuffer']
    idxs = np.zeros(cb.value.shape)
    for i in range(cb.value.shape[0]):
        idx = np.argmin(np.abs(rr.datetime[:].astype('datetime64[ms]') - cb.datetime[i].astype('datetime64[ms]')))
        idxs[i] = idx
    cb.rawdata = [rr]
    cb.rawdata_indices = idxs
    cb.method = station_info['widpro_method'] 
    cc = d.new(cb)

   
    # Read in the flux estimates for assumed height
    e3 = d.read(station_info['files']['flux_ah'],
                date='2016-11-01', ftype='minidoas-scan', timeshift=13)
    fb = e3['FluxBuffer']
    dt = fb.datetime[:].astype('datetime64[s]')
    indices = []
    for _dt in dt:
        idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]') - _dt))
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

    # Read in the flux estimates for calculated height
    e4 = d.read(station_info['files']['flux_ch'],
                date='2016-11-01', ftype='minidoas-scan',
                station=station_info['wp_station_id'], timeshift=13)
    fb1 = e4['FluxBuffer']
    dt = fb1.datetime[:].astype('datetime64[s]')
    indices = []
    for _dt in dt:
        idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]') - _dt))
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
        new_description = mb3.description[0] + '; plume geometry inferred from triangulation'
        mb3.description = new_description
        mb3.name = 'WS2PVT'
        m3 = d.new(mb3)
        
    gfb2 = e4['GasFlowBuffer']
    gfb2.methods = [m3]
    gf2 = d.new(gfb2)
    fb1.gasflow = gf2
    f1 = d.new(fb1)

    # Now read in preferred flux values for assumed height downloaded from FITS
    data_ah = np.loadtxt(station_info['files']['fits_flux_ah'],
                      dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                      skiprows=1, delimiter=',')
    dates = data_ah['date'].astype('datetime64[s]')
    indices = []
    for i, dt in enumerate(dates):
        idx = np.argmin(np.abs(f.datetime[:].astype('datetime64[s]') - dt))
        indices.append(idx)
    pfb = PreferredFluxBuffer(fluxes=[f],
                              flux_indices=[indices],
                              value=data_ah['val'],
                              value_error=data_ah['err'],
                              datetime=dates.astype(str))
    d.new(pfb)

    # Now read in preferred flux values for calculated height downloaded from FITS
    data_ch = np.loadtxt(station_info['files']['fits_flux_ch'],
                         dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                         skiprows=1, delimiter=',')
    dates = data_ch['date'].astype('datetime64[s]')
    indices = []
    for i, dt in enumerate(dates):
        idx = np.argmin(np.abs(f1.datetime[:].astype('datetime64[s]') - dt))
        indices.append(idx)
    pfb1 = PreferredFluxBuffer(fluxes=[f1],
                              flux_indices=[indices],
                              value=data_ch['val'],
                              value_error=data_ch['err'],
                              datetime=dates.astype(str))
    d.new(pfb1)


def flux_ah(pf, perror_thresh=0.5):
    """
    Compute the flux for assumed height.
    """

    f = pf.fluxes[0]
    c = f.concentration
    for fidx in range(f.value[:].size):
        idx0, idx1 = f.concentration_indices[fidx]
        so2 = c.value[idx0:idx1]
        r = c.rawdata
        ridx = c.rawdata_indices[idx0:idx1]
        angles = r[0].inc_angle[ridx]
        bearing = r[0].bearing[0]

        # Get the wind direction
        gf = f.gasflow
        gf_times = f.gasflow.datetime[:].astype('datetime64[s]')
        gf_idx = np.argmin(np.abs(gf_times - np.datetime64(f.datetime[fidx])))
        vx, vy = gf.vx[gf_idx], gf.vy[gf_idx]
        wd = vec2bearing(vx,vy)
        ws = np.sqrt(vx*vx + vy*vy)

        t_angle = np.cos(np.deg2rad(bearing - wd - 90.))

        _, _, elev0 = gf.position[gf_idx]
        _, _, elev2 = gf.position[gf_idx+2]
        plumewidth = elev2-elev0

        plmax = so2.max()
        edge = 0.05*plmax
        pidx = np.where(so2 >= edge)
        plstart = pidx[0][0]
        plend = pidx[0][-1]
        plrange = np.abs(np.deg2rad(angles[plstart]) - np.deg2rad(angles[plend]))
        # calculate distance between measurements assuming dx = r * theta
        int_time = r[0].integration_time[ridx]/1000.
        dx = np.ones(angles.size)*0.015707963*1000./992*int_time*plumewidth/plrange

        col_amt = dx[plstart:plend+1] * so2[plstart:plend+1]
        ica = np.sum(col_amt)

        # correct for a non-perpendicular transect through the plume
        ica = abs(t_angle * ica)
        ppmm_to_kgs = 2.660e-06
        kgs_to_tday = 86.4
        flux = ica * ws * 0.000230688
        # Compute percentage error
        x = pf.value[fidx]*86.4
        x0 = flux
        p_error = 100.*abs(x0-x)/x
        if p_error > perror_thresh:
            msg = "Error of {} exceeds threshold for assumed height flux.\n".format(p_error)
            msg += "Date: {}\n".format(f.datetime[fidx])
            msg += "Expected flux: {}; Estimated flux: {}.\n".format(x, x0)
            raise MDOASException(msg)
 

def flux_ch(pf_ch, pf_ah, perror_thresh=0.5):
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

    for i, dt in enumerate(f_ch.datetime[:].astype('datetime64[s]')):
        idx = np.argmin(np.abs(f_ah.datetime[:].astype('datetime64[s]') - dt))
        _, _, h = gf_ch.position[i*3+1]
        lon, lat, h1 = gf_ah.position[idx*3+1]
        idx0, idx1 = f_ah.concentration_indices[idx]
        idx_max = np.argmax(c_ah.value[idx0:idx1])
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
            msg = "Error of {} exceeds threshold for calculated height flux.\n".format(p_error)
            msg += "Date: {}\n".format(str(dt))
            msg += "Expected flux: {}; Estimated flux: {}.\n".format(x, x0)
            raise MDOASException(msg)
       

def verify_flux(filename, perror_thresh=0.5):
    """
    Verify that the recomputed flux values are close to the 
    flux values stored in FITS.
    """
    d = Dataset.open(filename)
    for instrument in ['WI301', 'WI302']:
        for _pf in d.elements['PreferredFlux']:
            gfm = _pf.fluxes[0].gasflow.methods[0].name[:]
            sid = _pf.fluxes[0].concentration.rawdata[0].instrument.name[:]
            if gfm == 'WS2PV' and sid == instrument:
                pf_ah = _pf
            elif gfm == 'WS2PVT' and sid == instrument:
                pf_ch = _pf
        try:
            flux_ah(pf_ah, perror_thresh=perror_thresh)
            flux_ch(pf_ch, pf_ah, perror_thresh=perror_thresh)
        except MDOASException, e:
            msg = str(e)
            msg += "File: {}\n".format(filename)
            raise MDOASException(msg) 
    d.close()


def is_file_OK(filename):
    """
    Check that file exist and contains more than
    just a header line.
    """

    if os.path.isfile(filename):
        with open(filename) as fh:
            linecount = len(fh.readlines())
        if linecount < 2:
            return False
    else:
        return False
    return True


def FITS_download(date, station, outputpath='/tmp'):
    st_lookup = {'WI301':'NE', 'WI302':'SR'}
    nztz = timezone('Pacific/Auckland')
    date_start = nztz.localize(datetime.datetime(date.year, date.month, date.day, 0, 0, 0))
    date_end = nztz.localize(datetime.datetime(date.year, date.month, date.day, 23, 59, 59))
    date_start_utc = date_start.astimezone(utc)
    date_end_utc = date_end.astimezone(utc)

    base_url="https://fits.geonet.org.nz/observation"
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
        df = pd.read_csv(url.format(base_url, station, method), 
                         index_col=0, parse_dates=True,
                         skiprows=1, names=['obs', 'error'])
        df_new = df.loc[(df.index>=date_start_utc) & (df.index <= date_end_utc)]
        if df_new.size < 1:
            filepath = None
        else:
            df_new.to_csv(filepath)
        filepaths.append(filepath)
    return filepaths


raw_data_path = '/home/yannik/GeoNet/minidoas/'
date = datetime.date(2017, 7, 22)
date = datetime.date(2016, 11, 1)
dates = [date]
for date in dates:
    outputfile = 'MiniDOAS_{:d}{:02d}{:02d}.h5'.format(date.year, date.month, date.day)
    outputpath = '/tmp'
    if True:
        d = Dataset(os.path.join(outputpath, outputfile), 'w')

        # ToDo: get correct plume coordinates
        tb = TargetBuffer(name='White Island main plume',
                          target_id='WI001',
                          position=[177.18375770, -37.52170799, 321.0])
        t = d.new(tb)
        
        wpoptions = "{'Pixel316nm':479, 'TrimLower':30, 'LPFilterCount':3, 'MinWindSpeed':3,"
        wpoptions += "'BrightEnough':500, 'BlueStep':5, 'MinR2:0.8, 'MaxFitCoeffError':50.0,"
        wpoptions += "'InPlumeThresh':0.05, 'MinPlumeAngle':0.1, 'MaxPlumeAngle':3.0,"
        wpoptions += "'MinPlumeSect':0.4, 'MaxPlumeSect':2.0, 'MeanPlumeCtrHeight':310,"
        wpoptions += "'SEMeanPlumeCtrHeight':0.442, 'MaxRangeToPlume':5000, 'MaxPlumeWidth':2600"
        wpoptions += "'MaxPlumeCentreAltitude':2000, 'MaxRangeSeperation':5000,"
        wpoptions += "'MaxAltSeperation':1000, 'MaxTimeDiff':30,"
        wpoptions += "'MinTriLensAngle':0.1745, 'MaxTriLensAngle':2.9671,"
        wpoptions += "'SEWindSpeed':0.20, 'WindMultiplier':1.24, 'SEWindDir':0.174}"
        mb1 = MethodBuffer(name='WidPro v1.2',
                           description='Jscript wrapper for DOASIS',
                           settings=wpoptions)
        m1 = d.new(mb1)

        station_info = {}
        station_info['WI301'] = {'files':{},
                                 'stationID': 'WI301',
                                 'stationLoc':'White Island North-East Point', 
                                 'target':t,
                                 'bearing':6.0214,
                                 'lon':177.192979384, 'lat':-37.5166903535, 'elev': 49.0,
                                 'widpro_method':m1,
                                 'wp_station_id':'NE'}

        station_info['WI302'] = {'files':{},
                                 'stationID': 'WI302',
                                 'stationLoc':'White Island South Rim', 
                                 'target':t,
                                 'bearing':3.8223,
                                 'lon':177.189013316, 'lat':-37.5265334424, 'elev':96.0,
                                 'widpro_method':m1,
                                 'wp_station_id':'SR'}

        for station in ['WI301', 'WI302']:
            
            # Find the raw data
            raw_data_filename = "{:s}_{:d}{:02d}{:02d}.zip".format(station_info[station]['wp_station_id'],
                                                                   date.year, date.month, date.day) 
            raw_data_filepath = os.path.join(raw_data_path, 'spectra',
                                             station_info[station]['wp_station_id'],
                                             raw_data_filename)
            if os.path.isfile(raw_data_filepath):
                with ZipFile(raw_data_filepath) as myzip:
                    myzip.extractall('/tmp')
                raw_data_filename = raw_data_filename.replace('.zip', '.csv')
                raw_data_filepath = os.path.join('/tmp', raw_data_filename)
            if not is_file_OK(raw_data_filepath):
                raw_data_filepath = None

            station_info[station]['files']['raw'] = raw_data_filepath
            
            # Find the concentration data
            monthdir = '{:d}-{:02d}'.format(date.year, date.month)
            spectra_filename = "{:s}_{:d}_{:02d}_{:02d}_Spectra.csv"
            spectra_filename = spectra_filename.format(station_info[station]['wp_station_id'],
                                                       date.year, date.month, date.day)
            spectra_filepath = os.path.join(raw_data_path, 'results',
                                            monthdir,
                                            spectra_filename)
            if not is_file_OK(spectra_filepath):
                spectra_filepath = None

            station_info[station]['files']['spectra'] = spectra_filepath
            
            # Find the flux data
            flux_ah_filename = spectra_filename.replace('Spectra.csv', 'Scans.csv')
            flux_ah_filepath = os.path.join(raw_data_path, 'results',
                                            monthdir,
                                            flux_ah_filename)
            if not is_file_OK(flux_ah_filepath):
                flux_ah_filepath = None
                
            station_info[station]['files']['flux_ah'] = flux_ah_filepath
     
            flux_ch_filename = "XX_{:d}_{:02d}_{:02d}_Combined.csv"
            flux_ch_filename = flux_ch_filename.format(date.year, date.month, date.day)
            flux_ch_filepath = os.path.join(raw_data_path, 'results',
                                            monthdir,
                                            flux_ch_filename)
            if not is_file_OK(flux_ch_filepath):
                flux_ch_filepath = None

            station_info[station]['files']['flux_ch'] = flux_ch_filepath

            fits_flux_ah, fits_flux_ch = FITS_download(date, station)
            station_info[station]['files']['fits_flux_ah'] = fits_flux_ah
            station_info[station]['files']['fits_flux_ch'] = fits_flux_ch


            read_single_station(d, station_info[station])

        # Wind data
        windd_dir = os.path.join(raw_data_path, 'wind', 'direction')
        winds_dir = os.path.join(raw_data_path, 'wind', 'speed')
        sub_dir = '{:02d}-{:02d}'.format(date.year-2000, date.month)
        winds_filename = '{:d}{:02d}{:02d}_WS_00.txt'.format(date.year,
                                                             date.month,
                                                             date.day)
        windd_filename = winds_filename.replace('WS', 'WD')
        winds_filepath = os.path.join(winds_dir, sub_dir, winds_filename)
        windd_filepath = os.path.join(windd_dir, sub_dir, windd_filename)
        
        if is_file_OK(winds_filepath) and is_file_OK(windd_filepath):
            # Read in the raw wind data; this is currently not needed to reproduce
            # flux estimates so it's just stored for reference
            e2 = d.read({'direction': windd_filepath,
                         'speed': winds_filepath},
                         ftype='minidoas-wind', timeshift=13)
            gfb = e2['GasFlowBuffer']
            gf = d.new(gfb)
                                                                 

        d.close()

    verify_flux(os.path.join(outputpath, outputfile), 1.)

