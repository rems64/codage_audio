from scipy.io.wavfile import read
import scipy.fft
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math

# wave = read("symboleA.wav")
# wave = read("symboleA2.wav")
# wave = read("symboleU.wav")


def get_frequency(signal, sample_rate, beta, width, order):
    N = signal.shape[0]
    M = 10*N

    t = np.arange(M)/sample_rate
    freq = scipy.fft.rfftfreq(t.shape[-1], d=1/sample_rate)

    sos = scipy.signal.butter(
        order, [501-width, 527+width], btype="bandpass", output='sos', fs=sample_rate)
    filtered = scipy.signal.sosfilt(sos, signal)

    window = np.kaiser(N, beta)
    # window = np.hamming(N)
    signal_ffted = np.abs(scipy.fft.rfft(window*filtered, M))
    # signal_ffted = np.abs(scipy.fft.rfft(window*signal, M))
    
    # plt.plot(freq, signal_ffted)
    # plt.plot(freq, signal_ffted_filtered)
    # plt.show()

    win_freq_min = 501
    win_freq_max = 527

    slicing_min = int(win_freq_min/freq[-1]*freq.shape[0])
    slicing_max = int(win_freq_max/freq[-1]*freq.shape[0])
    
    index_max = slicing_min + np.argmax(signal_ffted[slicing_min:slicing_max])
    freq_max = freq[index_max]
    value_max = signal_ffted[index_max]
    if value_max < 400_000:
        return None
    # print(f"value_max {signal_ffted[index_max]}")
    # print(f"value_acote {signal_ffted[index_max+1000]}")
    
    # plt.plot(freq[slicing_min:slicing_max], signal_ffted[slicing_min:slicing_max])
    # plt.yscale("log")
    # plt.show()

    return freq_max


def frequency_to_letter(freq: float):
    guess = chr(ord('A')+max(0, round(freq)-501))
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

def distance(a, b):
    return sum([1 for i in range(min(len(a), len(b))) if a[i] != b[i]])+(abs(len(a)-len(b)))

def decode_serie():
    # sample_rate, signal = read("mot3.wav")
    # sample_rate, signal = read("mess_ssespace.wav")
    sample_rate, signal = read("mess.wav")
    # sample_rate, signal = read("mess_difficile.wav")
    signal_length = signal.shape[0]
    symbol_length = 2000
    symbol_padding = 2500
    signals = []
    
    target = "C EST MIEUX COMME CA"
    # target = "CESTPASMAL"
    # print(f'{distance("C EST MIEUX COMME CA", "C EST MIEUX COMME CA")}')
    # print(f'{distance("C EST MIEUX CAMME CA", "C EST MIEUX COMME CA")}')

    for t in range(0, signal_length, symbol_padding):
        signals.append(signal[t:t+symbol_length])
    
    # Best: beta=4
    betas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10, 10.5, 11]
    # 45 is best
    widths = [10*i for i in range(-1, 10)]
    # Best is 
    orders = [4*i+2 for i in range(10)]
    
    print(f"Signal of {len(signals)} letters")
    best_score = math.inf
    for beta in betas:
        for width in widths:
            for order in orders:
                # print(f"beta={beta}; ", end='')
                # print(f"width={width};\t", end='')
                # print(f"order={order};\t", end='')
                letters = []
                for signal in signals:
                    if np.all(signal==0):
                        letters.append(" ")
                        # print(" ", end='')
                        continue
                    # freq = get_frequency(signal, sample_rate, 4, 70, order)
                    freq = get_frequency(signal, sample_rate, beta, width, order)
                    if freq is None:
                        letters.append(" ")
                        # print(" ", end='')
                        continue
                    guess = frequency_to_letter(freq)
                    letters.append(guess)
                guess = ''.join(letters)
                score = distance(target, guess)
                if score <= best_score:
                    best_score = score
                    # print(f"Best: {guess}, score: {score}")
                    print(f"beta={beta}; width={width}; order={order};\t", end='')
                    print(guess, end='')
                    print(f"\t{distance(target, guess)}")

# decode_singles()
decode_serie()