import pandas as pd
from tdmsdata_edited import TdmsData
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import doppler_shift_2025
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
                         'Approx Time': raw_data[:, 4], 'SDUMP': raw_data[:, 5]}) # laser freq is from the signal 

def get_idler_freq(df):
    pump_freq = 384.301267 # THz
    df['Idler Frequency'] = pump_freq - df['Laser Frequency (THz)']
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


# def doppler_shift(dataset, isotope): 
    # importlib.reload(doppler_shift_2025)
    # doppler_df = dataset.copy()
    # freq = doppler_df['Laser Frequency (THz)']
    # shifted_freq = doppler_shift_2025.getshift(freq, isotope)
    # doppler_df['Laser Frequency (THz)'] = shifted_freq
    # return doppler_df

def scale_ops(df, poly_degree=4):
    cycle_data = df.groupby('Laser Frequency (THz)')['Time (s)']
    sum_bycycle = cycle_data.sum()
    count_bycycle = cycle_data.count()
    avg_bycycle = sum_bycycle / count_bycycle
    freq_bycycle = np.array(avg_bycycle.index)

    # Weighted Polynomial Fit
    weights = count_bycycle  # Weighted by the number of times the frequency was measured
    X_poly = np.array(np.vander(avg_bycycle, N=poly_degree + 1, increasing=False))  # Polynomial features
    model = sm.WLS(freq_bycycle, X_poly, weights=weights).fit()
    fit_line = model.predict(X_poly)

    # Unweighted Polynomial Fit
    # unweighted_model = sm.OLS(freq_bycycle, X_poly).fit()
    # unweighted_fit_line = unweighted_model.predict(X_poly)

    # Get coefficients
    coeffs = model.params  # Polynomial coefficients
    r_value = model.rsquared  # Coefficient of determination (R^2)

    plt.figure()
    plt.scatter(df['Time (s)'], df['Laser Frequency (THz)'], s=1)
    plt.xlim(15,60)
    # plt.scatter(avg_bycycle, freq_bycycle, color='red', label='Average')
    # plt.plot(avg_bycycle, fit_line, color='red', label='WLS Fit (Weighted)')
    plt.show()

    return coeffs

def get_scale(df) -> list:
    cycle_scales = []
    grouped_by_cycle = df.groupby('Cycle No.')

    for cycle_no, cycle_df in grouped_by_cycle: # cycle is number, group is df of the cycle 
        coeffs = scale_ops(cycle_df)
        cycle_scales.append({'cycle': cycle_no, 'coefficients': coeffs.tolist()}) 
   
    return cycle_scales

def process_scaled_df(df, cycle_scales) ->  pd.DataFrame:
    bin_width = 50 # MHz 
    cycle_dfs = []

    # get a scaled df for each cycle
    for info_by_cycle in cycle_scales: # cycle_scales is a list of dictionaries 
        cycle_number = info_by_cycle['cycle']
        coeffs = info_by_cycle['coefficients']

        cycle_df = df[df['Cycle No.'] == cycle_number].copy() # filter for cycle
        scaled_cycle_df = cycle_df.assign(
            scaled_freq=lambda df: np.polyval(coeffs, df['Time (s)']) # for scaled data 
        )
        cycle_dfs.append(scaled_cycle_df)

    scaled_df = pd.concat(cycle_dfs, ignore_index=True)
    
    freq_range = (scaled_df['scaled_freq'].max() - scaled_df['scaled_freq'].min()) * 1e6 # MHz
    scaled_bins = int(np.ceil(freq_range / bin_width)) # number of bins will change to ensure bin worth is consistent 

    scaled_df = (
        scaled_df
        .assign(Freq_bin_scaled=lambda df: pd.cut(df['scaled_freq'], scaled_bins))
        .groupby('Freq_bin_scaled', observed=True) # separate counts for each bin and cycle
        .size()
        .reset_index(name='Count raw')
        .assign(Bin_center=lambda df: df['Freq_bin_scaled'].apply(lambda x: x.mid))
    )
    scaled_df.columns = ['Freq bin', 'Count', 'Bin center']

    scaled_df['Freq bin'] = scaled_df['Freq bin'].astype('category')  # Match global dtype
    scaled_df['Bin center'] = scaled_df['Bin center'].astype(float)
 
    return scaled_df 

def bin_events(df):
    freq = df['Laser Frequency (THz)'].to_numpy()
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
    centers = centers[2:-2]
    counts  = counts[2:-2]
    
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
        # doppler_df = doppler_shift_calc(clean_df, isotope)
        cycle_scales = get_scale(clean_df) # list of coefficients for freq interpolation 
        scaled_df = process_scaled_df(clean_df, cycle_scales)
        binned_df = bin_events(clean_df)
        print(clean_df)

        df = clean_df
        plt.figure()
        plt.scatter(df['Time (s)'], df['Laser Frequency (THz)'], s=1)
        plt.xlim(20,30)
        plt.ylim(423.349,423.3501)
        # plt.scatter(avg_bycycle, freq_bycycle, color='red', label='Average')
        # plt.plot(avg_bycycle, fit_line, color='red', label='WLS Fit (Weighted)')
        plt.show()

        yield binned_df, filename, isotope