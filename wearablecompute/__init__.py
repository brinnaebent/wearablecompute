import numpy as np
import pandas as pd
import datetime as datetime
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from scipy import signal
from scipy.integrate import trapz

'''
    Feature Engineering of Wearable Sensors:
     
    Metrics computed:
        Mean Heart Rate Variability
        Median Heart Rate Variability
        Maximum Heart Rate Variability
        Minimum Heart Rate Variability
        SDNN (HRV)
        RMSSD (HRV)
        NNx (HRV)
        pNNx (HRV)
        HRV Frequency Domain Metrics:
            PowerVLF
            PowerLF
            PowerHF
            PowerTotal
            LF/HF
            PeakVLF
            PeakLF
            PeakHF
            FractionLF
            FractionHF
        EDA Peaks
        Activity Bouts
        Interday Summary: 
            Interday Mean
            Interday Median
            Interday Maximum 
            Interday Minimum 
            Interday Quartile 1
            Interday Quartile 3
        Interday Standard Deviation 
        Interday Coefficient of Variation 
        Intraday Standard Deviation (mean, median, standard deviation)
        Intraday Coefficient of Variation (mean, median, standard deviation)
        Intraday Mean (mean, median, standard deviation)
        Daily Mean
        Intraday Summary:
            Intraday Mean
            Intraday Median
            Intraday Minimum
            Intraday Maximum
            Intraday Quartile 1
            Intraday Quartile 3
        TIR (Time in Range of default 1 SD)
        TOR (Time outside Range of default 1 SD)
        POR (Percent outside Range of default 1 SD)
        MASE (Mean Amplitude of Sensor Excursions, default 1 SD)
        Hours from Midnight (circadian rhythm feature)
        Minutes from Midnight (ciracadian rhythm feature)

        
    '''


def e4import(filepath, sensortype, Starttime='NaN', Endtime='NaN', window='5min'): #window is in seconds
    """
        brings in an empatica compiled file **this is not raw empatica data**
        Args:
            filepath (String): path to file
            sensortype (Sting): Options: 'EDA', 'HR', 'ACC', 'TEMP', 'BVP'
            Starttime (String): (optional, default arg = 'NaN') format '%Y-%m-%d %H:%M:%S.%f', if you want to only look at data after a specific time
            Endtime (String): (optional, default arg = 'NaN') format '%Y-%m-%d %H:%M:%S.%f', if you want to only look at data before a specific time
            window (String): default '5min'; this is the window your data will be resampled on.
        Returns:
            (pd.DataFrame): dataframe of data with Time, Mean, Std columns
    """
    
    if sensortype == 'ACC':
        data = pd.read_csv(filepath,header=None, names = ["Time", "x", "y", "z"]) 
        data['Var'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        data = data.drop(columns=['x', 'y', 'z'])   
    else:
        data = pd.read_csv(filepath, header=None, names=['Time', 'Var'])
        
    
    data['Time'] =  pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    
    if Starttime != 'NaN':
        VarData = data.loc[data.loc[:, 'Time'] >= Starttime, :]
        if Endttime != 'NaN':
            VarData = VarData.loc[VarData.loc[:, 'Time'] <= Endtime, :]
    else:
        VarData = data
    
    
    Data = pd.DataFrame()
    Data[[(sensortype + '_Mean')]] = VarData.resample(window, on='Time').mean()
    Data[[(sensortype + '_Std')]] = VarData.resample(window, on='Time').std()
    
    Data = Data.reset_index()
    
    print((sensortype + ' Import and Resample Complete'))
    
    return(Data)


def HRV(time, IBI, ibimultiplier = 1000):
    """
        computes Heart Rate Variability metrics
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            maxHRV (FloatType): maximum HRV
            minHRV (FloatType): minimum HRV
            meanHRV (FloatType): mean HRV
            medianHRV(FloatType): median HRV
    """
    time = time
    ibi = IBI*ibimultiplier
    
    maxHRV = round(max(ibi) * 10) / 10
    minHRV = round(min(ibi) * 10) / 10
    meanHRV = round(np.mean(ibi) * 10) / 10
    medianHRV = round(np.median(ibi) * 10) / 10
    
    return maxHRV, minHRV, meanHRV, medianHRV


def SDNN(time, IBI, ibimultiplier=1000):
    """
        computes Heart Rate Variability metric SDNN
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            SDNN (FloatType): standard deviation of NN intervals 
    """
    time = time
    ibi = IBI*ibimultiplier
    
    SDNN = round(np.sqrt(np.var(ibi, ddof=1)) * 10) / 10 
    
    return SDNN


def RMSSD(time, IBI, ibimultiplier=1000):
    """
        computes Heart Rate Variability metric RMSSD
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
        Returns:
            RMSSD (FloatType): root mean square of successive differences
            
    """
    time = time
    ibi = IBI*ibimultiplier
    
    differences = abs(np.diff(ibi))
    rmssd = np.sqrt(np.sum(np.square(differences)) / len(differences))
    
    return round(rmssd * 10) / 10


def NNx(time, IBI, ibimultiplier=1000, x=50):
    """
        computes Heart Rate Variability metrics NNx and pNNx
        Args:
            time (pandas.DataFrame column or pandas series): time column
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
            x (IntegerType): default = 50; set the number of times successive heartbeat intervals exceed 'x' ms
        Returns:
            NNx (FloatType): the number of times successive heartbeat intervals exceed x ms
            pNNx (FloatType): the proportion of NNx divided by the total number of NN (R-R) intervals. 
    """
    time = time
    ibi = IBI*ibimultiplier
    
    differences = abs(np.diff(ibi))
    n = np.sum(differences > x)
    p = (n / len(differences)) * 100
    
    return (round(n * 10) / 10), (round(p * 10) / 10)

def FrequencyHRV(IBI, ibimultiplier=1000, fs=1):
    """
        computes Heart Rate Variability frequency domain metrics
        Args:
            IBI (pandas.DataFrame column or pandas series): column with inter beat intervals
            ibimultiplier (IntegerType): defualt = 1000; transforms IBI to milliseconds. If data is already in ms, set as 1
            fs (IntegerType): Optional sampling frequency for frequency interpolation (default=1)
        Returns:
            (dictionary): dictionary of frequency domain HRV metrics with keys:
                PowerVLF (FloatType): Power of the Very Low Frequency (VLF): 0-0.04Hz band
                PowerLF (FloatType): Power of the Low Frequency (LF): 0.04-0.15Hz band
                PowerHF (FloatType): Power of the High Frequency (HF): 0.15-0.4Hz band
                PowerTotal (FloatType):Total power over all frequency bands
                LF/HF (FloatType): Ratio of low and high power
                Peak VLF (FloatType): Peak of the Very Low Frequency (VLF): 0-0.04Hz band
                Peak LF (FloatType): Peak of the Low Frequency (LF): 0.04-0.15Hz band
                Peak HF (FloatType): Peak of the High Frequency (HF): 0.15-0.4Hz band
                FractionLF (FloatType): Fraction that is low frequency
                FractionHF (FloatType): Fraction that is high frequency
                
    """
    
    ibi = IBI*ibimultiplier
    steps = 1 / fs

    # create interpolation function based on the rr-samples.
    x = np.cumsum(ibi) / 1000.0
    f = interp1d(x, ibi, kind='cubic')

    # sample from interpolation function
    xx = np.arange(1, np.max(x), steps)
    ibi_interpolated = f(xx)
    fxx, pxx = signal.welch(x=ibi_interpolated, fs=fs)

    '''
    Segement found frequencies in the bands 
    - Very Low Frequency (VLF): 0-0.04Hz 
    - Low Frequency (LF): 0.04-0.15Hz 
    - High Frequency (HF): 0.15-0.4Hz
    '''
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    # calculate power in each band by integrating the spectral density
    vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = trapz(pxx[cond_lf], fxx[cond_lf])
    hf = trapz(pxx[cond_hf], fxx[cond_hf])

    # sum these up to get total power
    total_power = vlf + lf + hf

    # find which frequency has the most power in each band
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

    # fraction of lf and hf
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)

    results = {}
    results['PowerVLF'] = round(vlf, 2)
    results['PowerLF'] = round(lf, 2)
    results['PowerHF'] = round(hf, 2)
    results['PowerTotal'] = round(total_power, 2)
    results['LF/HF'] = round(lf / hf, 2)
    results['PeakVLF'] = round(peak_vlf, 2)
    results['PeakLF'] = round(peak_lf, 2)
    results['PeakHF'] = round(peak_hf, 2)
    results['FractionLF'] = round(lf_nu, 2)
    results['FractionHF'] = round(hf_nu, 2)

    return results


def PeaksEDA(eda, time):
    """
        calculates peaks in the EDA signal
        Args:
            eda (pandas.DataFrame column or pandas series): eda column
            time (pandas.DataFrame column or pandas series): time column
        Returns:
            countpeaks (IntegerType): the number of peaks total 
            peakdf (pandas.DataFrame): a pandas dataframe with time and peaks to easily integrate with your data workflow
    """  
    
    EDAy = eda.to_numpy()
    EDAx = time.to_numpy()
    
    peaks, _ = find_peaks(EDAy, height=0, distance=4, prominence=0.3)
    
    peaks_x = []
    for i in peaks:
        px = time.iloc[i]
        peaks_x.append(px)
    
    peakdf = pd.DataFrame()
    peakdf['Time'] = peaks_x
    peakdf['Peak'] = ([1]*len(peaks_x))
    
    countpeaks = len(peakdf)

    return countpeaks, peakdf


def exercisepts(acc, hr, time): #acc and hr must be same length, acc must be magnitude
    """
        calculates activity bouts using accelerometry and heart rate
        Args:
            acc (pandas.DataFrame column or pandas series): accelerometry column
            hr (pandas.DataFrame column or pandas series): heart rate column
            time (pandas.DataFrame column or pandas series): time column
        Returns:
            countbouts (IntegerType): the number of acitvity bouts total
            returndf (pandas.DataFrame): a pandas dataframe with time and activity bouts (designated as a '1') to easily integrate with your data workflow
    """  
    
    exercisepoints = []
    for z in range(len(acc)):
        if acc[z] > np.mean(acc[0:z]):
            if hr[z] > np.mean(hr[0:z]):
                exercisepoints.append(1)
            else:
                exercisepoints.append(0)
        else:
            exercisepoints.append(0)
            
    returndf = pd.DataFrame()
    returndf['Time'] = time
    returndf['Activity Bouts'] = exercisepoints
    
    countbouts = len(exercisepoints)
    return countbouts, returndf


def interdaycv(column):
    """
        computes the interday coefficient of variation on pandas dataframe Sensor column
        Args:
            column (pandas.DataFrame column or pandas series): column that you want to calculate over
        Returns:
            cvx (IntegerType): interday coefficient of variation 
    """
    cvx = (np.std(column) / (np.nanmean(column)))*100
    return cvx


def interdaysd(column):
    """
        computes the interday standard deviation of pandas dataframe Sensor column
        Args:
            column (pandas.DataFrame column or pandas series): column that you want to calculate over
        Returns:
            interdaysd (IntegerType): interday standard deviation 
    """
    interdaysd = np.std(column)
    return interdaysd


def intradaycv(column, time, timeformat='%Y-%m-%d %H:%M:%S.%f'):
    """
        computes the intradaycv, returns the mean, median, and sd of intraday cv Sensor column in pandas dataframe
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            intradaycv_mean (IntegerType): Mean, Median, and SD of intraday coefficient of variation 
            intradaycv_median (IntegerType): Median of intraday coefficient of variation 
            intradaycv_sd (IntegerType): SD of intraday coefficient of variation 
        Requires:
            interdaycv() function
    """
    intradaycv = []
    
    df = pd.DataFrame()
    df['Column'] = column
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Day'] = df['Time'].dt.date
    
    for i in pd.unique(df['Day']):
        intradaycv.append(interdaycv(df[df['Day']==i]['Column']))
    
    intradaycv_mean = np.mean(intradaycv)
    intradaycv_median = np.median(intradaycv)
    intradaycv_sd = np.std(intradaycv)
    
    return intradaycv_mean, intradaycv_median, intradaycv_sd


def intradaysd(column, time, timeformat='%Y-%m-%d %H:%M:%S.%f'):
    """
        computes the intradaysd, returns the mean, median, and sd of intraday sd Sensor column in pandas dataframe
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            intradaysd_mean (IntegerType): Mean, Median, and SD of intraday standard deviation 
            intradaysd_median (IntegerType): Median of intraday standard deviation 
            intradaysd_sd (IntegerType): SD of intraday standard deviation 
    """
    intradaysd =[]
    
    df = pd.DataFrame()
    df['Column'] = column
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Day'] = df['Time'].dt.date
    
    for i in pd.unique(df['Day']):
        intradaysd.append(np.std(df[df['Day']==i]['Column']))
    
    intradaysd_mean = np.mean(intradaysd)
    intradaysd_median = np.median(intradaysd)
    intradaysd_sd = np.std(intradaysd)
    
    return intradaysd_mean, intradaysd_median, intradaysd_sd


def intradaymean(column, time, timeformat='%Y-%m-%d %H:%M:%S.%f'):
    """
        computes the intradaymean, returns the mean, median, and sd of the intraday mean of the Sensor data
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            intradaysd_mean (IntegerType): Mean, Median, and SD of intraday standard deviation of glucose
            intradaysd_median (IntegerType): Median of intraday standard deviation of glucose
            intradaysd_sd (IntegerType): SD of intraday standard deviation of glucose
    """
    intradaymean =[]
    
    df = pd.DataFrame()
    df['Column'] = column
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Day'] = df['Time'].dt.date
    
    for i in pd.unique(df['Day']):
        intradaymean.append(np.nanmean(df[df['Day']==i]['Column']))
    
    intradaymean_mean = np.mean(intradaymean)
    intradaymean_median = np.median(intradaymean)
    intradaymean_sd = np.std(intradaymean)

    return intradaymean_mean, intradaymean_median, intradaymean_sd

def dailymean(column, time, timeformat='%Y-%m-%d %H:%M:%S.%f'):
    """
        computes the mean of each day
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            pandas.DataFrame with days and means as columns
        
    """
    
    intradaymean =[]
    
    df = pd.DataFrame()
    df['Column'] = column
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Day'] = df['Time'].dt.date
    
    for i in pd.unique(df['Day']):
        intradaymean.append(np.nanmean(df[df['Day']==i]['Column']))
        
    dailymeandf = pd.DataFrame()
    dailymeandf['Day'] = pd.unique(df['Day'])
    dailymeandf['Mean'] = intradaymean
        
    return dailymeandf


def dailysummary(column, time, timeformat='%Y-%m-%d %H:%M:%S.%f'):
    """
        computes the summary of each day (mean, median, std, max, min, Q1G, Q3G)
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            pandas.DataFrame with days and summary metrics as columns
        
    """
    
    intradaymean =[]
    intradaymedian =[]
    intradaysd =[]
    intradaymin =[]
    intradaymax =[]
    intradayQ1 =[]
    intradayQ3 =[]
    
    df = pd.DataFrame()
    df['Column'] = column
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Day'] = df['Time'].dt.date
    
    for i in pd.unique(df['Day']):
        intradaymean.append(np.nanmean(df[df['Day']==i]['Column']))
        intradaymedian.append(np.nanmedian(df[df['Day']==i]['Column']))
        intradaysd.append(np.std(df[df['Day']==i]['Column']))
        intradaymin.append(np.nanmin(df[df['Day']==i]['Column']))
        intradaymax.append(np.nanmax(df[df['Day']==i]['Column']))
        intradayQ1.append(np.nanpercentile(df[df['Day']==i]['Column'], 25))
        intradayQ3.append(np.nanpercentile(df[df['Day']==i]['Column'], 75))
        
    dailysumdf = pd.DataFrame()
    dailysumdf['Day'] = pd.unique(df['Day'])
    dailysumdf['Mean'] = intradaymean
    dailysumdf['Median'] = intradaymedian
    dailysumdf['Standard Deviation'] = intradaysd
    dailysumdf['Minimum'] = intradaymin
    dailysumdf['Maximum'] = intradaymax
    dailysumdf['Quartile 1'] = intradayQ1
    dailysumdf['Quartile 3'] = intradayQ3
        
    return dailysumdf


def interdaysummary(column, dataframe=True):
    """
        computes interday mean, median, minimum and maximum, and first and third quartile over a column
        Args:
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             dataframe (True/False): default=True; whether you want a pandas DataFrame as an output or each of the summary metrics as IntegerTypes
        Returns:
            pandas.DataFrame with columns: Mean, Median, Standard Deviation, Minimum, Maximum, Quartile 1, Quartile 3
            
            or
            
            interdaymean (FloatType): mean 
            interdaymedian (FloatType): median 
            interdaysd (FloatType) : standard deviation
            interdaymin (FloatType): minimum 
            interdaymax (FloatType): maximum 
            interdayQ1 (FloatType): first quartile 
            interdayQ3 (FloatType): third quartile 
    """
    
    interdaymean = np.nanmean(column)
    interdaymedian = np.nanmedian(column)
    interdaysd = np.std(column)
    interdaymin = np.nanmin(column)
    interdaymax = np.nanmax(column)
    interdayQ1 = np.nanpercentile(column, 25)
    interdayQ3 = np.nanpercentile(column, 75)
    
    interdaysum = pd.DataFrame()
    interdaysum['Mean'] = [interdaymean]
    interdaysum['Median'] = interdaymedian
    interdaysum['Standard Deviation'] = interdaysd
    interdaysum['Minimum'] = interdaymin
    interdaysum['Maximum'] = interdaymax
    interdaysum['Quartile 1'] = interdayQ1
    interdaysum['Quartile 3'] = interdayQ3
    
    if dataframe == True:
        return interdaysum
    else:
        return interdaymean, interdaymedian, interdaysd, interdaymin, interdaymax, interdayQ1, interdayQ3


def TIR(df, column, sd=1, sr=1):
    """
        computes time in the range of (default=1 sd from the mean)column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): sampling rate of sensor
        Returns:
            TIR (IntegerType): Time in Range set by sd, *Note time is relative to your SR
            
    """
    up = np.mean(column) + sd*np.std(column)
    dw = np.mean(column) - sd*np.std(column)
    TIR = len(df[(column <= up) & (column >= dw)])*sr 
    return TIR


def TOR(df, column, sd=1, sr=1):
    """
        computes time outside the range of (default=1 sd from the mean) column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): sampling rate of sensor
        Returns:
            TOR (IntegerType): Time outside of range set by sd, *Note time is relative to your SR
    """
    up = np.mean(column) + sd*np.std(column)
    dw = np.mean(column) - sd*np.std(column)
    TOR = len(df[(column >= up) | (column <= dw)])*sr 
    return TOR

def POR(df, column, sd=1, sr=1):
    """
        computes percent time outside the range of (default=1 sd from the mean) column in pandas dataframe
        Args:
             df (pandas.DataFrame):
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             sd (IntegerType): standard deviation from mean for range calculation (default = 1 SD)
             sr (IntegerType): 
        Returns:
            POR (IntegerType): percent of time spent outside range set by sd
    """
    up = np.mean(column) + sd*np.std(column)
    dw = np.mean(column) - sd*np.std(column)
    TOR = len(df[(column >= up) | (column <= dw)])*sr 
    POR = (TOR/(len(column)*sr))*100
    return POR

def MASE(df, column, sd=1):
    """
        computes the mean amplitude of sensor excursions (default = 1 sd from the mean)
        Args:
             df (pandas.DataFrame):
             column (pandas.DataFrame column or pandas series): column that you want to calculate over
             sd (IntegerType): standard deviation from mean to set as a sensor excursion (default = 1 SD)
        Returns:
           MASE (IntegerType): Mean Amplitude of sensor excursions
    """
    up = np.mean(column) + sd*np.std(column)
    dw = np.mean(column) - sd*np.std(column)
    MASE = np.mean(df[(column >= up) | (column <= dw)])
    return MASE

def crhythm(time, timeformat='%Y-%m-%dT%H:%M:%S'):
    """
        computes 'minutes from midnight' and 'hours from midnight'- these features will allow you to account for circaidan rhythm effects
        Args:
             time (pandas.DataFrame): time column
             timeformat (String): default = '%Y-%m-%d %H:%M:%S.%f'; format of timestamp in time column
        Returns:
            hourfrommid (ListType): Hours from midnight, the same length as your time column
            minfrommid (ListType): Minutes from midnight, the same length as your time column

    """
    
    df = pd.DataFrame()
    df['Time'] =  pd.to_datetime(time, format=timeformat)
    df['Timefrommidnight'] =  df['Time'].dt.time
    
    hourfrommid=[]
    minfrommid=[]
    
    for i in range(0, len(df['Timefrommidnight'])):
        minfrommid.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))
        hourfrommid.append(round((int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2])*60 + int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9])/60))/60))

    return hourfrommid, minfrommid
