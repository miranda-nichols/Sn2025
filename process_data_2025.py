import pandas as pd
from tdmsdata_edited import TdmsData
import matplotlib.pyplot as plt
import numpy as np
import os
from natsort import natsorted
import importlib
import doppler_shift_2025

isotope_mapping = {
    'Sn_120': 120,
    'Sn_122': 122,
    'Sn_124': 124,
    'Sn_116': 116,
    'Sn_118': 118,
    'Sn_112': 112,
    'Sn_114': 114,
    'Sn_115': 115,
    'Sn_117': 117,
    'Sn_119': 119
}

def read_tdms(folder_path, file, channel):
    file_path = folder_path / file 
    TDMS = TdmsData(file_path)
    raw_data = TDMS.get_raw_data(f"Channel {channel}")  # 2D np array
    return raw_data

def create_df(raw_data):
    return pd.DataFrame({'Cycle No.': raw_data[:, 0], 'Time (s)': raw_data[:, 1], 
                         'Laser Frequency (THz)': raw_data[:, 2], 'Power (mW)': raw_data[:, 3], 
                         'Approx Time': raw_data[:, 4], 'SDUMP': raw_data[:, 5]})

# def doppler_shift(dataset, isotope): 
    # importlib.reload(doppler_shift_2025)
    # doppler_df = dataset.copy()
    # freq = doppler_df['Laser Frequency (THz)']
    # measured_voltage = doppler_df['LE Probe']
    # shifted_freq = doppler_shift_2025.getshift(freq, isotope, measured_voltage)
    # doppler_df['Laser Frequency (THz)'] = shifted_freq
    # return doppler_df

def process_tdms(raw_df):
    step_size = 0.001 # nm 
    bin_width_thz = 0.00008 
    # bins = 50

    # wavelengths_raw = raw_df['Laser Frequency (THz)']
    # wavelengths_raw = pd.to_numeric(wavelengths_raw, errors='coerce')
    # wavelengths_raw = wavelengths_raw.dropna()

    # # counts_df = pd.DataFrame({'Freq': wavelengths_raw})
    # # counts_df = wavelengths_raw.value_counts().reset_index()
    # # counts_df.columns = ['Freq', 'Count']
        
    # # counts_df = counts_df.sort_values(by='Freq', ascending=True).reset_index()

    # raw_df['Freq bin'] = pd.cut(raw_df['Laser Frequency (THz)'], bins, right=False)
    # # raw_df = raw_df[raw_df['Freq bin'] >= 400]

    # binned_df = raw_df.groupby('Freq bin', observed=True).size().reset_index(name='Count')
    # binned_df['Bin center'] = binned_df['Freq bin'].apply(lambda x: x.mid)
    # binned_df = binned_df[binned_df['Count'] > 0]
    # binned_df.columns = ['Freq bin', 'Count', 'Bin center']

    df = raw_df.copy()
    # df = df[df['Time (s)'] <= 90].reset_index(drop=True)
    df['Laser Frequency (THz)'] = pd.to_numeric(df['Laser Frequency (THz)'], errors='coerce')
    df = df.dropna(subset=['Laser Frequency (THz)'])
    df = df[df['Laser Frequency (THz)'] > 423.349].reset_index(drop=True)

    fmin = df['Laser Frequency (THz)'].min()
    fmax = df['Laser Frequency (THz)'].max()
    start = np.floor(fmin / bin_width_thz) * bin_width_thz
    stop  = np.ceil(fmax / bin_width_thz) * bin_width_thz + bin_width_thz
    bins = np.arange(start, stop + 0.5*bin_width_thz, bin_width_thz)

    df['Freq bin'] = pd.cut(df['Laser Frequency (THz)'], bins, right=False)  
    binned_df = (df
                 .groupby('Freq bin', observed=True)
                 .size()
                 .reset_index(name='Count'))

    # binned_df = binned_df[binned_df['Count'] > 0]
    binned_df['Bin center'] = binned_df['Freq bin'].apply(lambda iv: iv.mid)

    return binned_df

def main(folder_path): 
    for tdms_path in natsorted(folder_path.glob("*.tdms"), key=lambda p: p.name):
        filename = tdms_path.name
        isotope = next((value for key, value in isotope_mapping.items() if key in filename), None)
        raw_data = read_tdms(folder_path, filename, channel=1) 
        raw_df = create_df(raw_data)
        # doppler_df = doppler_shift_calc(raw_df, isotope)
        binned_df = process_tdms(raw_df)
        yield binned_df, filename, isotope