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

def noise_handling(raw_df, noise_df):
    signal_df = raw_df

    # clean freq col dropping anything that is non numeric 
    signal_df = signal_df.dropna(subset=['Laser Frequency (THz)'])

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

# def doppler_shift(dataset, isotope): 
    # importlib.reload(doppler_shift_2025)
    # doppler_df = dataset.copy()
    # freq = doppler_df['Laser Frequency (THz)']
    # shifted_freq = doppler_shift_2025.getshift(freq, isotope, measured_voltage)
    # doppler_df['Laser Frequency (THz)'] = shifted_freq
    # return doppler_df

def bin_events(df):
    df = df[df['Laser Frequency (THz)'] > 0.0].reset_index(drop=True)
    freq = df['Laser Frequency (THz)'].to_numpy()
    bin_width = 6e-5

    # Build uniform grid and histogram
    fmin, fmax = freq.min(), freq.max()
    start = np.floor(fmin / bin_width) * bin_width
    stop  = np.ceil(fmax  / bin_width) * bin_width + bin_width
    edges = np.arange(start, stop, bin_width)

    counts, edges = np.histogram(freq, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Drop edge bins (inj + end)
    centers = centers[1:-1]
    counts  = counts[1:-1]
    
    binned_df = pd.DataFrame({'Bin center': centers, 'Count': counts})
    return binned_df, edges

def bin_per_cycle(df, bin_edges):
    by_cycle = df.groupby('Cycle No.')
    nbins = len(bin_edges) - 1
    acc = np.zeros(nbins)
    n_cycles = 0

    for _, g in by_cycle:
        freq = g['Laser Frequency (THz)'].to_numpy()
        counts, _ = np.histogram(freq, bins=bin_edges)
        acc += counts
        n_cycles += 1

    mean_counts = acc / max(n_cycles, 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

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
      
        preprocessed_df = noise_handling(raw_df, noise_df=ch0_info)
        # doppler_df = doppler_shift_calc(raw_df, isotope)
        binned_df, edges = bin_events(preprocessed_df)
        bin_per_cycle_df = bin_per_cycle(preprocessed_df, edges)

        yield binned_df, bin_per_cycle_df, filename, isotope