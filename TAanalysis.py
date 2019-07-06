import numpy as np
import scipy as sc
import scipy.fftpack
from collections import deque
import CustomPrincetonSPE_v2 as SPE
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice, tee
from numba import jit
import datetime as dt
import time as ti


def binData(data, pixel_energy, bin_size, start_col=None, end_col=None):
    """Takes in data that is row averaged as a 2D array. Uses pixel to energy
    conversion along with given bin size to bin data."""

    if (start_col is not None) and (end_col is not None):
        print("Clipping Cols")
        pixel_energy = pixel_energy[start_col:end_col]
        print("Now %d" % len(pixel_energy))
        data = data[:,start_col:end_col]
    energyBins = np.arange(int(min(pixel_energy)), int(max(pixel_energy)) + 1,
                           bin_size)
    print("Made bins from %d,%d,%g" % (int(pixel_energy[0]),
                                       int(pixel_energy[-1]) + 1, bin_size))
    data_temp = np.zeros([len(data), np.size(energyBins)])
    for frame in range(len(data)):
        data_temp[frame] = np.interp(energyBins, pixel_energy[::-1],
                                     data[frame][::-1])
    data = data_temp
    return energyBins, data

#@jit(nopython=False)
def collectSPE(data_sets, time_file, comment, xmin, xmax, scatter):
    """Collects all SPE files together into numpy array by time point.
    Returns both pump on and pump off shots. Need to collect in dictionaries
    since files could have different number of frames and I need to collect
    frames by time point."""

    times = np.genfromtxt(time_file)
    num_times = len(times)
    f = 0
    pump_on_total = {time: [] for time in range(num_times)}
    pump_off_total = {time: [] for time in range(num_times)}
    for data_file in data_sets:

        data = loadSPE(data_file)
        pump_on = data[::2,:]
        pump_off = data[1::2,:]
        pump_on_times = []
        pump_off_times = []
        for time in range(num_times):
            pump_on_times.append(pump_on[time::num_times])
            pump_off_times.append(pump_off[time::num_times])
            if f == 0:
                pump_on_total[time] = pump_on_times[time]
                pump_off_total[time] = pump_off_times[time]
            else:
                #Need to append so I collect all frames belonging to same time pt
                pump_on_total[time] = np.append(pump_on_total[time],
                                                pump_on_times[time], axis=0)
                pump_off_total[time] = np.append(pump_off_total[time],
                                                 pump_off_times[time], axis=0)
        f += 1
    #Convert back to numpy array for further use
    pump_on_total = np.asarray([pump_on_total[time] for time in range(num_times)])
    pump_off_total = np.asarray([pump_off_total[time] for time in range(num_times)])

    return pump_on_total, pump_off_total

def prepareTA(data_sets, time_file, comment, xmin=None, xmax=None, scatter_files=None):
    """Overall function used for averaging SPE files together for transient
    absorption. Can trim x axis first. Can take pump scatter files and subtract
    them frame by frame for each pixel. Saves TA as npy file."""

    times = np.loadtxt(time_file)
    num_times = len(times)
    print('Collecting data files')
    scatter_on = None
    scatter_off = None
    pump_on, pump_off = collectSPE(data_sets, time_file, comment, xmin, xmax,
                                   scatter=False)
    if scatter_files is not None:
        print('Collecting scatter files')
        scatter_on, scatter_off = collectSPE(scatter_files, time_file, comment,
                                              xmin, xmax, scatter=True)

    p_on_clean = avgCollectedSPE(pump_on, num_times, 'on', xmin, xmax, scatter_on)
    p_off_clean = avgCollectedSPE(pump_off, num_times, 'off', xmin, xmax, scatter_off)

    p_on_clean_avg = np.mean(p_on_clean, axis=1)
    p_off_clean_avg = np.mean(p_off_clean, axis=1)

    dA = np.log(p_on_clean_avg/p_off_clean_avg)
    print('Saving dA file')
    np.save('%s_dA' %comment, dA)

#@jit(nopython=False)
def avgCollectedSPE(shot_file, num_times, on_off, xmin=None, xmax=None,
                    scatter_file = None, save_shot = False):
    """Uses median and MAD as robust estimators of each pixel to filter data.
    Can subtract pump scatter from each pixel at each frame. Can save pump on
    or off shot separately if you want."""

    print('There are %d times!'%num_times)
    print('Robust averaging pump %s shots!' %on_off)
    shot = shot_file

    if scatter_file is not None:
        scatter = scatter_file
        print('Subtracting pump scatter!')
    if xmin is not None and xmax is not None:
        shot = np.array([shot[time][:,:,xmin:xmax] for time in range(num_times)])
        if scatter_file is not None:
            scatter = np.array([scatter[time][:,:,xmin:xmax]
                                for time in range(num_times)])

    num_y = len(shot[0][0])
    num_x = len(shot[0][0][0])
    clean_shot = np.zeros((num_times, num_y, num_x))

    for time in range(num_times):
        for y_coord in range(num_y):
            for x_coord in range(num_x):
                pixel = shot[time][:,y_coord][:,x_coord]
                d_pixel = np.abs(pixel - np.median(pixel))
                MAD = np.median(d_pixel)
                z = d_pixel/(MAD if MAD else 1.) #modified Z score
                clean_pixel = pixel[z<2.]
                robust_avg = np.mean(clean_pixel)
                clean_shot[time][y_coord][x_coord] = robust_avg

        if scatter_file is not None:
            clean_shot[time] = clean_shot[time] - np.mean(scatter[time], axis=0)

        ts = ti.time()
        st = dt.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        print('Time: %d' %time, st)

    if save_shot is True:
        np.save('clean_%s' %on_off, clean_shot)

    return clean_shot

def FFTFilter(data, low_cut = None, high_cut = None, order = None, fs = None):
    """Fast fourier transform filter for """
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = sc.signal.butter(order, [low, high], btype = 'stop')
    w, h = sc.signal.freqs(b, a)
    y = sc.signal.lfilter(b, a, data)
    plt.figure(103)
    plt.plot(w, np.log10(abs(h)))
    plt.xscale('log')
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which = 'both', axis = 'both')
    plt.show()
    return y

def groundStateAbsOneFile(data_filename, rows, pixel_energy, comment,
                          bckg_file = None, hamp = 0, t = 0, mean = 0,
                          bin_size = 0, xmin = None, xmax = None, sig=None,
                          low_cut = 0, high_cut = 0, order = 0, u=None,
                          window_size = None, ymin = None, ymax = None, raw=False,
                          compare = False, save_harm = False):
    """Takes in SPE file for the ground state data, where even frames are
    on blank and odd frames are on sample. Can take in background SPE file.
    Can choose which rows to look at. Give it the pixel to energy conversion.
    Comment is title of plot. Can give it area of interest to plot."""

    data = loadSPE(data_filename)
    energy_axis = pixel_energy

    print('Summing over rows!')
    row_sum_data = np.sum(data[:, rows, :], 1)

    if bckg_file is not None:
        bckg = loadSPE(bckg_file)
        row_sum_bckg = np.sum(bckg[:, rows, :], 1)

    if bin_size > 0:
        print('Binning data!')
        energy_axis, row_sum_data = binData(row_sum_data, pixel_energy, bin_size)
        if bckg_file is not None:
            energy_axis, row_sum_bckg = binData(row_sum_bckg, pixel_energy, bin_size)

    samp = row_sum_data[1::2]
    blank = row_sum_data[::2]
    num_sets = len(samp)
    print('There are %d sets!' % num_sets)
    samp_avg = np.mean(samp, 0)
    blank_avg = np.mean(blank, 0)
    if bckg_file is not None:
        print('Subtracting background scatter!')
        bckg_samp = np.mean(row_sum_bckg[1::2], 0)
        bckg_blank = np.mean(row_sum_bckg[::2], 0)
        samp_avg -= bckg_samp
        blank_avg -= bckg_blank

    print('Calculating absorption!')
    dA = -np.log10(samp_avg/blank_avg)

    if order > 0:
        print('Fourier filtering absorption!')
        print('high_cut_max %d' %(len(energy_axis)/2))
        dAF = FFTFilter(dA, low_cut, high_cut, order, len(energy_axis))
    if hamp > 0:
        print('Hampel filtering absorption!')
        dAH = hampelFilt(dA, t, hamp)
        if order > 0:
            dAFH = hampelFilt(dAF, t, hamp)
    if mean > 0:
        print('Mean filtering absorption!')
        dAM = rollingMean(dA, mean)
        if order > 0:
            dAFM = rollingMean(dAF, mean)
        if hamp > 0:
            dAHM = rollingMean(dAH, mean)
            if order > 0:
                dAFHM = rollingMean(dAFH, mean)

    if save_harm is True:
        np.save('Harmonics_samp_%s' %comment, samp_avg)
        np.save('Harmonics_blank_%s' %comment, blank_avg)

    idxmin = (np.abs(energy_axis-xmin)).argmin()
    xmin = energy_axis[idxmin]
    idxmax = (np.abs(energy_axis-xmax)).argmin()
    xmax = energy_axis[idxmax]
    print(xmin, xmax)

    np.savetxt('%s_dA.txt' %comment, dAHM[idxmax:idxmin])

    print('Here is your plot!')
    plt.figure(101)

    if compare is True:

        plt.plot(energy_axis, dA, label = 'No filter')
        if order > 0:
            plt.plot(energy_axis, dAF, label = 'Fourier filter only')
        if hamp > 0:
            plt.plot(energy_axis, dAH, label = 'Hampel filter only')
            if order > 0:
                plt.plot(energy_axis, dAFH, label = 'Fourier + Hampel filter')
        if mean > 0:
            plt.plot(energy_axis, dAM, label = 'Mean filter only')
            if order > 0:
                plt.plot(energy_axis, dAFM, label = 'Fourier + mean filter')
            if hamp > 0:
                plt.plot(energy_axis, dAHM, label = 'Hampel + median filter')
        if (hamp > 0 and mean > 0) and (order > 0):
            plt.plot(energy_axis, dAFHM, label = 'Fourier + Hampel + meadian filter')

    else:
        if hamp == 0 and mean == 0 and order == 0:
            plt.plot(energy_axis, dA, label = 'No filter')
        if order > 0:
            plt.plot(energy_axis, dAF, label = 'Fourier filter only')
        if hamp > 0 and mean == 0:
            plt.plot(energy_axis, dAH, label = 'Hampel filter only')
        if mean > 0 and hamp == 0:
            plt.plot(energy_axis, dAM, label = 'Mean filter only')
        if mean > 0 and hamp == 0 and order > 0:
            plt.plot(energy_axis, dAFM, label = 'Fourier + mean filter')
        if hamp > 0 and mean == 0 and order > 0:
            plt.plot(energy_axis, dAFH, label = 'Fourier + Hampel filter')
        if hamp > 0 and mean > 0:
            plt.plot(energy_axis, dAHM, label = 'Hampel + mean filter')
        if (hamp > 0 and mean > 0) and (order > 0):
            plt.plot(energy_axis, dAFHM, label = 'Fourier + Hampel + mean filter')

    plt.title('Ground State Absorption of %s' % (comment))
    plt.xlabel('Energy (eV)')
    plt.ylabel('Absobance (OD)')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()

def groundStateAbsTwoFile(samp_file, blank_file, rows, pixel_energy, comment,
                          bckg_samp_file = None, bckg_blank_file = None,
                          bin_size = 0, xmin = None, xmax = None, sig=None,
                          low_cut = 0, high_cut = 0, order = 0, u=None,
                          window_size = None, ymin = None, ymax = None, raw=False,
                          compare = False, hamp = 0, t = 0, mean = 0):
    """Takes in SPE file for the ground state data, where even frames are
    on blank and odd frames are on sample. Can take in background SPE file.
    Can choose which rows to look at. Give it the pixel to energy conversion.
    Comment is title of plot. Can give it area of interest to plot."""

    samp = loadSPE(samp_file)
    blank = loadSPE(blank_file)
    num_frames = len(samp)
    data = np.append(samp, blank, axis = 0)
    energy_axis = pixel_energy

    print('Summing over rows!')
    row_sum_data = np.sum(data[:, rows, :], 1)

    if bckg_samp_file is not None and bckg_blank_file is not None:
        bckg_samp = loadSPE(bckg_samp_file)
        bckg_blank = loadSPE(bckg_blank_file)
        bckg = np.append(bckg_samp, bckg_blank)
        row_sum_bckg = np.sum(bckg[:, rows, :], 1)

    if bin_size > 0:
        print('Binning data!')
        energy_axis, row_sum_data = binData(row_sum_data, pixel_energy, bin_size)
        if bckg_samp_file is not None and bckg_blank_file is not None:
            energy_axis, row_sum_bckg = binData(row_sum_bckg, pixel_energy, bin_size)

    samp = row_sum_data[:num_frames]
    blank = row_sum_data[num_frames:]
    num_sets = len(samp)
    print('There are %d sets!' % num_sets)
    samp_avg = np.mean(samp, 0)
    blank_avg = np.mean(blank, 0)
    if bckg_samp_file is not None and bckg_blank_file is not None:
        print('Subtracting background scatter!')
        bckg_samp = np.mean(row_sum_bckg[::2], 0)
        bckg_blank = np.mean(row_sum_bckg[1::2], 0)
        samp_avg -= bckg_samp
        blank_avg -= bckg_blank

    print('Calculating absorption!')
    dA = -np.log10(samp_avg/blank_avg)

    if order > 0:
        print('Fourier filtering absorption!')
        print('high_cut_max %d' %(len(energy_axis)/2))
        dAF = FFTFilter(dA, low_cut, high_cut, order, len(energy_axis))
    if hamp > 0:
        print('Hampel filtering absorption!')
        dAH = hampelFilt(dA, t, hamp)
        if order > 0:
            dAFH = hampelFilt(dAF, t, hamp)
    if mean > 0:
        print('Mean filtering absorption!')
        dAM = rollingMean(dA, mean)
        if order > 0:
            dAFM = rollingMean(dAF, mean)
        if hamp > 0:
            dAHM = rollingMean(dAH, mean)
            if order > 0:
                dAFHM = rollingMean(dAFH, mean)

    np.save('Harmonics_samp_%s' %comment, samp_avg)
    np.save('Harmonics_blank_%s' %comment, blank_avg)

    print('Here is your plot!')
    plt.figure(101)

    if compare is True:

        plt.plot(energy_axis, dA, label = 'No filter')
        if order > 0:
            plt.plot(energy_axis, dAF, label = 'Fourier filter only')
        if hamp > 0:
            plt.plot(energy_axis, dAH, label = 'Hampel filter only')
            if order > 0:
                plt.plot(energy_axis, dAFH, label = 'Fourier + Hampel filter')
        if mean > 0:
            plt.plot(energy_axis, dAM, label = 'Mean filter only')
            if order > 0:
                plt.plot(energy_axis, dAFM, label = 'Fourier + mean filter')
            if hamp > 0:
                plt.plot(energy_axis, dAHM, label = 'Hampel + median filter')
        if (hamp > 0 and mean > 0) and (order > 0):
            plt.plot(energy_axis, dAFHM, label = 'Fourier + Hampel + meadian filter')

    else:
        if hamp == 0 and mean == 0 and order == 0:
            plt.plot(energy_axis, dA, label = 'No filter')
        if order > 0:
            plt.plot(energy_axis, dAF, label = 'Fourier filter only')
        if hamp > 0 and mean == 0:
            plt.plot(energy_axis, dAH, label = 'Hampel filter only')
        if mean > 0 and hamp == 0:
            plt.plot(energy_axis, dAM, label = 'Mean filter only')
        if mean > 0 and hamp == 0 and order > 0:
            plt.plot(energy_axis, dAFM, label = 'Fourier + mean filter')
        if hamp > 0 and mean == 0 and order > 0:
            plt.plot(energy_axis, dAFH, label = 'Fourier + Hampel filter')
        if hamp > 0 and mean > 0:
            plt.plot(energy_axis, dAHM, label = 'Hampel + mean filter')
        if (hamp > 0 and mean > 0) and (order > 0):
            plt.plot(energy_axis, dAFHM, label = 'Fourier + Hampel + mean filter')

    plt.title('Ground State Absorption of %s' % (comment))
    plt.xlabel('Energy (eV)')
    plt.ylabel('Absobance (OD)')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()

def hampelFilt(data, t, n, zeros = False):
    """Takes in 1D array of data and applies a generalized Hampel Filter to
    remove outliers. Assumes a Gaussian distribution of the data in order to
    use S values as a robust estimator of the standard deviation. t is a tuning
    parameter that returns a median filter if t = 0. n is the window size.
    For 'zeros' see movingWindow."""

    numSets = len(data)
    medians = []
    S = []
    med_filt = np.zeros(numSets)
    med_windows, dev_windows1, dev_windows2 = tee(movingWindow(data, n, zeros), 3)
    for i in range(numSets):
        medians.append(np.median(next(med_windows)))
        S.append(1.4826 * np.median(abs(np.median(next(dev_windows1)) -
                                        next(dev_windows2))))
        if abs(data[i] - medians[i]) <= t * S[i]:
            med_filt[i] = data[i]
        elif abs(data[i] - medians[i]) > t * S[i]:
            med_filt[i] = medians[i]

    return med_filt

def loadSPE(filename, rows = None):
    """Takes in a an SPE file and returns it as a 3D numpy array with
    dims(frames, rows, columns)."""

    loaded_spe = SPE.PrincetonSPEFile(filename, rows)
    data = loaded_spe.getData(rows = rows)
    return data

def movingWindow(iterable, n, zeros = False):
    """Takes a 1D iterable and yields moving windows, with size n, centered at
    every element of the iterable. To keep a fixed window size, pads the edges
    with the edge values by default. Can also pad with 0's instead."""

    k = (n - 1) // 2
    it = iter(iterable)
    win = deque(islice(it, k + 1))
    for i in range(k):
        if zeros == False:
            win.appendleft(iterable[0])
        else:
            win.appendleft(0)
    yield win
    for elem in range(len(iterable) - 1):
        try:
            win.popleft()
            win.append(iterable[elem + (k + 1)])
        except IndexError:
            if zeros == False:
                win.append(iterable[-1])
            else:
                win.append(0)
        yield win

def rollingMean(data, n):
    """Takes in 1D array of data and applies a rolling mean filter."""

    num_sets = len(data)
    mean_filt = np.zeros(num_sets)
    mean_windows = movingWindow(data, n)
    for i in range(num_sets):
        mean_filt[i] = np.mean(next(mean_windows))
    return mean_filt

def workupTransient(avg_file, time_file, pixel_energy, comment, bin_size = 0,
                    thresh = 0.0005, scale = 0.0005, cstride = 3, rstride = 3,
                    p = 1, color = 'RdYlBu', time_zero = None, timeslices = None,
                    order = None, low_cut = 0, high_cut = 0,
                    xmin = None, xmax = None, sub_bckg = 0, hamp = 0, mean = 0,
                    t = 0, energyslices = None, semilog = False, plot3D = False,
                    avg_timeslices = False, n = None, avg_time = False,
                    save_TA = False):
    """Takes in averaged data file (npy), time point file (txt),
    energy calibration file (txt), smooths, and plots data. Can save the dA
    output as npy file."""

    energy_axis = pixel_energy[:,1]
    dA = np.load(avg_file)
    times = np.genfromtxt(time_file)
    if time_zero is not None:
        times -= time_zero
    num_times = len(times)
    if bin_size > 0:
        energy_axis, dA = binData(dA, energy_axis, bin_size)
        energy_axis = energy_axis[::-1]
    dAfilt = []
    append = dAfilt.append
    if low_cut > 0 and high_cut > 0:
        print('\nApplying fourier filter!')
        dA = FFTFilter(dA, low_cut, high_cut, order, len(energy_axis))
    if sub_bckg > 0:
        print('\nSubtracting background time points!')
        dA -= np.mean(dA[:sub_bckg], 0)
    if hamp == 0 and mean > 0:
        print('\nApplying rolling mean only!')
        for time in range(num_times):
            append(rollingMean(dA[time], mean))
    elif hamp > 0 and mean == 0:
        print('\nApplying Hampel filter only!')
        for time in range(num_times):
            append(hampelFilt(dA[time], t, hamp))
    elif hamp > 0 and mean > 0:
        print('\nApplying Hampel filter and then rolling mean!')
        for time in range(num_times):
            append(hampelFilt(dA[time], t, hamp))
            dAfilt[time] = rollingMean(dAfilt[time],mean)
    if hamp > 0 or mean > 0:
        dA = dAfilt
    dA = np.array(dA)
    idxmin = (np.abs(energy_axis-xmin)).argmin()
    xmin = energy_axis[idxmin]
    idxmax = (np.abs(energy_axis-xmax)).argmin()
    xmax = energy_axis[idxmax]
    #In case the data file is already trimmed
    num_bins = len(dA[0])
    if num_bins < 1340:
        Z = dA
        diff = (num_bins - (idxmin - idxmax))/2
        if diff % 2 == 0:
            idxmin += int(diff)
            idxmax -= int(diff)
        else:
            idxmin += int(np.ceil(diff))
            idxmax -= int(np.floor(diff))
        energy_axis = energy_axis[idxmax:idxmin]
    else:
        Z = dA[:,idxmax:idxmin]
        energy_axis = energy_axis[idxmax:idxmin]
    X, Y = np.meshgrid(energy_axis, times)
    if avg_time is True:
        time_avg = [0 for time_point in range(num_times)]
        for time_point in range(num_times):
            time_avg[time_point] = np.sum(Z[time_point-n:time_point+n],0)/(2*n)
            if time_point == 0:
                n*Z[0] + Z
                time_avg[time_point] = np.sum(Z[:time_point+n],0)/(2*n)
            if time_point == -1:
                Z + n*Z[-1]
                time_avg[time_point] = np.sum(Z[time_point-n:],0)/(2*n)
        Z = np.asarray(time_avg)
    print('\nMaking your plots!')
    plotTA(energy_axis, times, X=X, Y=Y, Z=Z, semilog=semilog, plot3D=plot3D,
           xmax=xmax, xmin=xmin, energyslices=energyslices, timeslices=timeslices,
           thresh=thresh, scale=scale, cstride=cstride, rstride=rstride, color=color)
    if save_TA is True:
        workup_TA = np.c_[times, Z]
        workup_TA = np.r_[[np.insert(energy_axis, 0, 0)], workup_TA]
        np.save('%s_workup.npy' %comment, workup_TA)
        print ('File Saved!')

def plotTA(energies, times, wav, X=None, Y=None, Z=None, semilog=False, plot3D=False,
           xmin=None, xmax=None, energyslices=None, timeslices=None, thresh=None,
           scale=None, cstride=None, rstride=None, color=None, contour=False,
           totals=None, init=None, fin=None, coeff1=None, coeff2=None):
    """Can plot time slices, energy slices and 3D plot for workup function.
    Can also plot two state model coefficients."""

    if energyslices is not None:
        if semilog:
            times -= -1000
        for energyslice in energyslices:
            idslice = (np.abs(energies-energyslice)).argmin()
            x = times
            y = Z[:, idslice]
            plt.figure('Energyslices')
            if semilog:
                plt.semilogx(x, y, label = '%.2f eV' % energies[idslice])
            else:
                plt.plot(x, y, label = '%.2f eV' % energies[idslice])
        plt.xlabel('Time (fs)')
        plt.ylabel('$\Delta$A')
        plt.legend()
        plt.show()

    if timeslices is not None:
        for lineout in timeslices:
            idtime = (np.abs(times-lineout)).argmin()
            x = energies
            y = Z[idtime]
            plt.figure('Timeslices')
            plt.plot(x, y, label = '%d fs' % times[idtime])
        plt.xlabel('Energy (eV)')
        plt.ylabel('$\Delta$A')
        plt.legend()
        plt.show()

    if plot3D:
        fig = plt.figure('3D')
        ax = Axes3D(fig)
        if semilog:
            times -= 1000
            Y = np.log10(Y)
        norm = mp.colors.SymLogNorm(linthresh = thresh, linscale = scale,
                                    vmin = -1, vmax = 1, clip=False)
        ax.plot_surface(X, Y, Z, cmap = getattr(mp.cm, color), cstride = cstride,
                    rstride = rstride, norm=norm, linewidth=0, antialiased=False)
        ax.set_xlim3d(xmin, xmax)
        plt.show()

    if contour:
        plt.figure('Contour %s' %wav)
        if semilog:
            times -= -1000
            ax = plt.axes()
            ax.set_yscale('log')
            ax.set_ylim(1000, 3e3)
            plt.contourf(energies, times, totals, cmap='RdYlBu', levels=100)
        else:
            plt.contourf(energies, times, totals, cmap='RdYlBu', levels=100)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Time (fs)')

    if init is not None and fin is not None:
        plt.figure('Timeslices %s' %wav)
        ax = plt.axes()
        ax.plot(energies, init*1000, label = 'Charge Transfer State')
        ax.plot(energies, fin*1000, label = 'Polaron State')
        plt.xlabel('Energy (eV)')
        plt.ylabel('$\Delta$ Abs. (mOD)')
        plt.legend(loc='upper right')

    if coeff1 and coeff2:
        plt.figure('Coeff %s' %wav)
        if semilog:
            ax = plt.axes()
            plt.semilogx(times, coeff1, 'o')
            plt.semilogx(times, coeff2, 'o')
        else:
            plt.scatter(times, coeff1)
            plt.scatter(times, coeff2)
        plt.xlabel('Time (fs)')
        plt.show()