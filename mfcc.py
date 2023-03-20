import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

def blackman_harris_window(N):
    """
    Create a Blackman-Harris Window
    
    Parameters
    ----------
    N: int
        Length of window
    
    Returns
    -------
    ndarray(N): Samples of the window
    """
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    t = np.arange(N)/N
    return a0 - a1*np.cos(2*np.pi*t) + a2*np.cos(4*np.pi*t) - a3*np.cos(6*np.pi*t)

def stft(x, w, h, win_fn=blackman_harris_window):
    """
    Compute the complex Short-Time Fourier Transform (STFT)
    Parameters
    ----------
    x: ndarray(N)
        Full audio clip of N samples
    w: int
        Window length
    h: int
        Hop length
    win_fn: int -> ndarray(N)
        Window function
    
    Returns
    -------
    ndarray(w, nwindows, dtype=np.complex) STFT
    """
    N = len(x)
    nwin = int(np.ceil((N-w)/h))+1
    # Make a 2D array
    # The rows correspond to frequency bins
    # The columns correspond to windows moved forward in time
    S = np.zeros((w, nwin), dtype=np.complex)
    # Loop through all of the windows, and put the fourier
    # transform amplitudes of each window in its own column
    for j in range(nwin):
        # Pull out the audio in the jth window
        xj = x[h*j:h*j+w]
        # Zeropad if necessary
        if len(xj) < w:
            xj = np.concatenate((xj, np.zeros(w-len(xj))))
        # Apply window function
        xj = win_fn(w)*xj
        # Put the fourier transform into S
        S[:, j] = np.fft.fft(xj)
    return S

def amplitude_to_db(S, amin=1e-10, ref=1):
    """
    Convert an amplitude spectrogram to be expressed in decibels
    
    Parameters
    ----------
    S: ndarray(win, T)
        Amplitude spectrogram
    amin: float
        Minimum accepted value for the spectrogram
    ref: int
        0dB reference amplitude
        
    Returns
    -------
    ndarray(win, T)
        The dB spectrogram
    """
    SLog = 20.0*np.log10(np.maximum(amin, S))
    SLog -= 20.0*np.log10(np.maximum(amin, ref))
    return SLog

def get_mel_filterbank(K, w, sr, min_freq, max_freq, n_bins):
    """
    Compute a mel-spaced filterbank
    
    Parameters
    ----------
    K: int
        Number of non-redundant frequency bins
    w: int
        Window length (should be around 2*K)
    sr: int
        The sample rate, in hz
    min_freq: int
        The center of the minimum mel bin, in hz
    max_freq: int
        The center of the maximum mel bin, in hz
    n_bins: int
        The number of mel bins to use
    
    Returns
    -------
    ndarray(n_bins, K)
        The triangular mel filterbank
    """
    bins = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins+2)*w/sr
    bins = np.array(np.round(bins), dtype=int)
    Mel = np.zeros((n_bins, K))
    for i in range(n_bins):
        i1 = bins[i]
        i2 = bins[i+1]
        if i1 == i2:
            i2 += 1
        i3 = bins[i+2]
        if i3 <= i2:
            i3 = i2+1
        tri = np.zeros(K)
        tri[i1:i2] = np.linspace(0, 1, i2-i1)
        tri[i2:i3] = np.linspace(1, 0, i3-i2)
        Mel[i, :] = tri
    return Mel

def get_dct_basis(NDCT, N):
    """
    Return a DCT Type-III basis

    Parameters
    ----------
    NDCT: int
        Number of DCT basis elements
    N: int
        Number of samples in signal
    
    Returns
    -------
    B: ndarray(NDCT, N)
        An NDCT x N matrix of DCT basis
    """
    ts = np.arange(1, 2*N, 2)*np.pi/(2.0*N)
    fs = np.arange(1, NDCT)
    B = np.zeros((NDCT, N))
    B[1::, :] = np.cos(fs[:, None]*ts[None, :])*np.sqrt(2.0/N)
    B[0, :] = 1.0/np.sqrt(N)
    return B

def mfcc(x, sr, w, h, n_bands=40, f_max=8000, n_mfcc=20, lifterexp = 0):
    """
    Compute MFCC Features

    Parameters
    ----------
    x: ndarray(N)
        A flat array of audio samples
    sr: int
        Sample rate
    w: int
        Window length
    h: int
        Hop length
    n_bands: int
        Number of mel bands to use
    f_max: int
        Maximum frequency
    n_mfcc: int
        Number of MFCC coefficients to return
    lifterexp: int
        Lifter exponential
    
    Returns
    -------
    ndarray(n_mfcc, n_windows)
        An array of MFCC samples
    """
    ## TODO: Fill this in
    ## Step 1: Compute the spectrogram
    S = stft(x, w, h)

    ## Step 2: Convert STFT to a mel-spaced spectrogram
    
    ## Step 3: Get log amplitude of the Mel-spaced spectrogram

    ## Step 4: Do DCT on each column of XMel

