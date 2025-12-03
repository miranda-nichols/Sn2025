import pandas as pd
import importlib
from tdmsdata_edited import TdmsData
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import doppler_shift_2025
importlib.reload(doppler_shift_2025)
import statsmodels.api as sm

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
                         'Approx Time': raw_data[:, 4], 'SDUMP': raw_data[:, 5]}) # laser freq is from the doubled signal 

def get_idler_freq(df):
    pump_freq = 384.301267 # THz
    df['Idler Frequency'] = pump_freq - 0.5*df['Laser Frequency (THz)']
    return df

def clean_data(signal_df, noise_df): # removes na and freqs<0, cycles with noise, and adds idler freq column
    # clean freq col dropping anything that is non numeric 
    valid = signal_df['Laser Frequency (THz)'].notna() & (signal_df['Laser Frequency (THz)'] > 0)
    signal_df = signal_df[valid].reset_index(drop=True)
    signal_df = get_idler_freq(signal_df)

    # electrical noise handling by removing entire cycle
    if noise_df is not None and not noise_df.empty:
        # cycles that have noise (from noise_df)
        noisy_cycles = (
            noise_df
            .dropna(subset=['Cycle No.'])
            ['Cycle No.']
            .unique()
        )
        # drop entire noisy cycles from signal_df
        mask = ~signal_df['Cycle No.'].isin(noisy_cycles)
        df_clean = signal_df.loc[mask].reset_index(drop=True)
        # print(f"Noise in {len(noisy_cycles)} cycles: {noisy_cycles}")
        return df_clean
    return signal_df

def doppler_shift(df, isotope): 
    freq = df['Idler Frequency']
    shifted_freq = doppler_shift_2025.getshift(freq, isotope)
    df['Idler Frequency'] = shifted_freq
    return df

def bin_events(df):
    freq = df['Idler Frequency'].to_numpy()
    bin_width = 7e-5

    # Build uniform grid and histogram
    fmin, fmax = freq.min(), freq.max()
    start = np.floor(fmin / bin_width) * bin_width
    stop  = np.ceil(fmax  / bin_width) * bin_width + bin_width
    edges = np.arange(start, stop, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)

    counts, edges = np.histogram(freq, bins=edges)

    # Drop edge bins (inj + end)
    centers = centers[1:-2]
    counts  = counts[1:-2]
    
    binned_df = pd.DataFrame({'Bin center': centers, 'Count': counts})
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
      
        clean_df = clean_data(raw_df, noise_df=ch0_info) # df with noise removed 
        doppler_df = doppler_shift(clean_df, isotope)
        binned_df = bin_events(doppler_df)

        yield binned_df, filename, isotope