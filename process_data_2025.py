import pandas as pd
import importlib
from tdmsdata_edited import TdmsData
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import doppler_shift_2025
importlib.reload(doppler_shift_2025)
import statsmodels.api as sm
from scipy.cluster.vq import kmeans2

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
    all_cycles = signal_df['Cycle No.'].dropna().unique()

    # electrical noise handling by removing entire cycle
    if noise_df is not None and not noise_df.empty:
        # cycles that have noise (from noise_df)
        noisy_cycles = (
            noise_df
            .dropna(subset=['Cycle No.'])
            ['Cycle No.']
            .unique()
        )

         # If ALL cycles are noisy → keep everything
        if set(noisy_cycles) >= set(all_cycles):
            return signal_df

        # drop entire noisy cycles from signal_df
        mask = ~signal_df['Cycle No.'].isin(noisy_cycles)
        df_clean = signal_df.loc[mask].reset_index(drop=True)
        # print(f"Noise in {len(noisy_cycles)} cycles: {noisy_cycles}")
        # print(all_cycles)
        return df_clean
    return signal_df

def drop_lower_branch(df, freq_col='Laser Frequency (THz)'):
    """
    Permanently remove the lower-frequency scan branch.
    Works even if the two scans overlap slightly.
    """

      # x = index (or use a real x-column if you have one)
    x = np.arange(len(df))
    y = df[freq_col].to_numpy().astype(float)

    # 2D points: [index, frequency]
    data = np.column_stack((x, y))

    # k-means into 2 clusters (two scans)
    centers, labels = kmeans2(data, k=2, minit='points')

    # find which cluster has the LOWER mean frequency
    mean_freq = [y[labels == k].mean() for k in range(2)]
    lower_label = int(np.argmin(mean_freq))

    # drop the lower-frequency cluster
    mask = labels != lower_label
    return df[mask].reset_index(drop=True)

def doppler_shift(df, isotope): 
    freq = df['Idler Frequency']
    shifted_freq = doppler_shift_2025.getshift(freq, isotope)
    df['Idler Frequency'] = shifted_freq
    return df

def bin_events(df, n_bins=120, smooth_window=3):
    freq = df[f'Idler Frequency'].to_numpy()
    freq = freq[np.isfinite(freq)]

    fmin, fmax = freq.min(), freq.max()
    edges = np.linspace(fmin, fmax, n_bins + 1)
    # widths = np.diff(edges)

    # simple histogram (no weights yet!)
    counts, edges = np.histogram(freq, bins=edges)
    centers = 0.5*(edges[:-1] + edges[1:])

    # optional smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        counts = np.convolve(counts, kernel, mode="same")
    
    # Drop edge bins (inj + end) and bins with no counts 
    mask = counts > 0
    centers = centers[mask][1:-3]
    counts  = counts[mask][1:-3]

    binned_df = pd.DataFrame({'Bin center': centers, 'Count': counts})
    return binned_df


# def bin_events(df):
#     freq = df['Idler Frequency'].to_numpy()
#     bin_width = 7e-5

#     # Build uniform grid and histogram
#     fmin, fmax = freq.min(), freq.max()
#     start = np.floor(fmin / bin_width) * bin_width
#     stop  = np.ceil(fmax  / bin_width) * bin_width + bin_width
#     edges = np.arange(start, stop, bin_width)
#     centers = 0.5 * (edges[:-1] + edges[1:])
#     n_bins = len(centers)
#     print(n_bins)

#     counts, edges = np.histogram(freq, bins=edges)

#     counts = _smooth_counts(counts.astype(float))

#     # Drop edge bins (inj + end) and bins with no counts 
#     mask = counts > 0
#     centers = centers[mask][1:-3]
#     counts  = counts[mask][1:-3]
    
#     binned_df = pd.DataFrame({'Bin center': centers, 'Count': counts})

def get_scale(df):
    grouped_by_cycle = df.groupby('Cycle No.')

    for cycle in grouped_by_cycle: # cycle is number, group is df of the cycle 
        plt.figure()
        plt.scatter(df['Time (s)'], df['Laser Frequency (THz)'], s=1)
        plt.show()


def main(folder_path): 
    for tdms_path in natsorted(folder_path.glob("*.tdms"), key=lambda p: p.name):
        filename = tdms_path.name
        isotope = next((value for key, value in isotope_mapping.items() if key in filename), None)
        
        # Read signal channel
        raw_data = read_tdms(folder_path, filename, channel=1) 
        raw_df = create_df(raw_data)

        if isotope == 116:
            raw_df = drop_lower_branch(raw_df)

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

        # if isotope == 116:
        #     get_scale(clean_df)

        yield binned_df, filename, isotope