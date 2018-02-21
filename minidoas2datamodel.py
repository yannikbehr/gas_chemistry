import datetime
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pytz import timezone, utc

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.minidoas import MiniDoasException
from spectroscopy.visualize import plot
from spectroscopy.datamodel import (PreferredFluxBuffer,
                                    InstrumentBuffer,
                                    TargetBuffer,
                                    MethodBuffer)
from spectroscopy.util import vec2bearing, bearing2vec


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



