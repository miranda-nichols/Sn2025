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

# def has_noise(noise_indicator):
#     return (noise_indicator > 0).any() # returns true if any entry > 0 from channel 0

def preprocess_events(raw_df, noise_df, frac_inj_cut=0.15):
    signal_df = raw_df.copy()

    # clean freq col by converting to numeric and dropping anything that is non numeric 
    signal_df['Laser Frequency (THz)'] = pd.to_numeric(signal_df['Laser Frequency (THz)'], errors='coerce')
    signal_df = signal_df.dropna(subset=['Laser Frequency (THz)'])

    # remove injection region by cutting first x% of points
    signal_df = signal_df.sort_values('Laser Frequency (THz)').reset_index(drop=True)
    n_cut = int(len(signal_df) * frac_inj_cut)
    signal_df = signal_df.iloc[n_cut:]

    # electrical noise handling 
    if noise_df is not None and not noise_df.empty:
        #### Removing common rows between channels 0 and 1 ####
        # # reliable indicators for both signal_df and noise_df
        # join_cols = [
        #     col for col in ('Cycle No.', 'Time (s)', 'Approx Time')
        #     if col in signal_df.columns and col in noise_df.columns
        # ]

        # # keep unique noise events for the join keys
        # noise_keys = noise_df[join_cols].dropna().drop_duplicates()

        # df_filtered = signal_df.merge(
        #     noise_keys.assign(_noise_hit=True),
        #     on=join_cols,
        #     how='left'
        # )
    
        # noise_mask = df_filtered['_noise_hit'] == True
        # df_removed = df_filtered[noise_mask].drop(columns=['_noise_hit'])
        # df_clean = df_filtered[~noise_mask].drop(columns=['_noise_hit'])

        # # print(noise_keys)
        # # print("Removed rows:")
        # # print(df_removed)
        # # print(signal_df)
        # print('Noise in:')
        # print(noise_df['Cycle No.'].nunique())
    
        # return df_clean

        #### Removing entire cycle if noise occurs ####.      
        # cycles that have noise (from noise_df)
        noisy_cycles = (
            noise_df
            .dropna(subset=['Cycle No.'])
            ['Cycle No.']
            .unique()
        )

        # print which cycles had noise
        print(f"Noise in {len(noisy_cycles)} cycles: {noisy_cycles}")

        # drop entire noisy cycles from signal_df
        df_clean = (
            signal_df[~signal_df['Cycle No.'].isin(noisy_cycles)]
            .reset_index(drop=True)
        )
        return df_clean
         
    return signal_df

# def doppler_shift(dataset, isotope): 
    # importlib.reload(doppler_shift_2025)
    # doppler_df = dataset.copy()
    # freq = doppler_df['Laser Frequency (THz)']
    # shifted_freq = doppler_shift_2025.getshift(freq, isotope, measured_voltage)
    # doppler_df['Laser Frequency (THz)'] = shifted_freq
    # return doppler_df

def bin_events(df):
    step_size = 0.001 # nm 
    bin_width_thz = 0.00008 

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

    binned_df['Bin center'] = binned_df['Freq bin'].apply(lambda iv: iv.mid)

    return binned_df

def main(folder_path): 
    for tdms_path in natsorted(folder_path.glob("*.tdms"), key=lambda p: p.name):
        filename = tdms_path.name
        isotope = next((value for key, value in isotope_mapping.items() if key in filename), None)

        # Read signal channel
        raw_data = read_tdms(folder_path, filename, channel=1) 
        raw_df = create_df(raw_data)

        # Read noise channel
        try:
            ch0 = read_tdms(folder_path, filename, channel=0)
            ch0_info = create_df(ch0)
        except Exception:
            # channel 0 missing or unreadable → no noise info
            ch0 = None
            ch0_info = None
      
        preprocessed_df = preprocess_events(raw_df, noise_df=ch0_info)
        # doppler_df = doppler_shift_calc(raw_df, isotope)
        binned_df = bin_events(preprocessed_df)

        yield binned_df, filename, isotope