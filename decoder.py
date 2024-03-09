from scipy.io.wavfile import read
import scipy.fft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# wave = read("symboleA.wav")
# wave = read("symboleA2.wav")
# wave = read("symboleU.wav")


def get_frequency(signal, sample_rate):
    N = signal.shape[0]
    M = 40*N

    t = np.arange(M)/sample_rate
    freq = scipy.fft.rfftfreq(t.shape[-1], d=1/sample_rate)

    sos = scipy.signal.butter(
        2, [480, 540], btype="bandpass", output='sos', fs=sample_rate)
    filtered = scipy.signal.sosfilt(sos, signal)

    window = np.hamming(N)
    signal_ffted = np.abs(scipy.fft.rfft(window*filtered, M))

    win_freq_min = 501
    win_freq_max = 527

    slicing_min = int(win_freq_min/freq[-1]*freq.shape[0])
    slicing_max = int(win_freq_max/freq[-1]*freq.shape[0])

    freq_max = freq[slicing_min +
                    np.argmax(signal_ffted[slicing_min:slicing_max])]
    
    # plt.plot(freq[slicing_min:slicing_max], signal_ffted[slicing_min:slicing_max])
    # plt.yscale("log")
    # plt.show()

    return freq_max


def frequency_to_letter(freq: float):
    guess = chr(ord('A')+max(0, int(freq)-501))
    return guess


def decode_singles():
    symbols = [("A", "A", "symboleA.wav"),
               ("A2", "A", "symboleA2.wav"),
               ("U", "U", "symboleU.wav"),
               ("U2", "U", "symboleU2.wav"),
               ("JUL1", "?", "Symbole_jules.wav"),
               ("JUL2", "?", "Symbole_jules2.wav")
               ]

    for symbol in symbols:
        id, expected, file = symbol
        sample_rate, signal = read(file)

        freq = get_frequency(signal, sample_rate)
        guess = frequency_to_letter(freq)

        print(f"[{id}]\tExpected: {expected}, guess: {guess}, freq_max: {freq}")

def decode_serie():
    sample_rate, signal = read("mot3.wav")
    # sample_rate, signal = read("mess_ssespace.wav")
    # sample_rate, signal = read("mess.wav")
    # sample_rate, signal = read("mess_difficile.wav")
    signal_length = signal.shape[0]
    symbol_length = 2000
    symbol_padding = 2500
    signals = []

    for t in range(0, signal_length, symbol_padding):
        signals.append(signal[t:t+symbol_length])
    
    print(f"Signal of {len(signals)} letters")
    
    for signal in signals:
        if np.all(signal==0):
            print(" ", end='')
            continue
        freq = get_frequency(signal, sample_rate)
        guess = frequency_to_letter(freq)
        print(guess, end='')

# decode_singles()
decode_serie()