from scipy.io.wavfile import read
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# wave = read("symboleA.wav")
# wave = read("symboleA2.wav")
# wave = read("symboleU.wav")
symbols = [("A", "A", "symboleA.wav"),
           ("A2", "A", "symboleA2.wav"),
           ("U", "U", "symboleU.wav"),
           ("U2", "U", "symboleU2.wav")]

def get_letter(signal, sample_rate):
    N = signal.shape[0]
    M = 40*N
    
    t = np.arange(M)/sample_rate
    freq = scipy.fft.rfftfreq(t.shape[-1], d=1/sample_rate)

    sos = scipy.signal.butter(2, [480, 540], btype="bandpass", output='sos', fs=sample_rate)
    filtered = scipy.signal.sosfilt(sos, signal)

    window = np.hamming(N)
    signal_ffted = np.abs(scipy.fft.rfft(window*filtered, M))

    win_freq_min = 501
    win_freq_max = 527

    slicing_min = int(win_freq_min/freq[-1]*freq.shape[0])
    slicing_max = int(win_freq_max/freq[-1]*freq.shape[0])

    freq_max = freq[slicing_min+np.argmax(signal_ffted[slicing_min:slicing_max])]
    guess = chr(65+max(0, int(freq_max)-501))
    return guess
    

for symbol in symbols:
    id, expected, file = symbol
    wave = read(file)
    sample_rate = wave[0]
    # print(f"sample_rate is {sample_rate}")
    signal = np.array(wave[1], dtype=float)
    N = signal.shape[0]
    # print(f"Duration is {N/sample_rate}")
    # signal = np.sin(100*2*np.pi*np.arange(N)/sample_rate)

    # print(signal)

    # M = (10**1) * N
    M = 40*N
    
    t = np.arange(M)/sample_rate
    freq = scipy.fft.rfftfreq(t.shape[-1], d=1/sample_rate)

    # signal_ffted = rfft(signal, N)

    sos = scipy.signal.butter(2, [480, 540], btype="bandpass", output='sos', fs=sample_rate)
    filtered = scipy.signal.sosfilt(sos, signal)

    window = np.hamming(N)
    signal_ffted = np.abs(scipy.fft.rfft(window*filtered, M))

    win_freq_min = 501
    win_freq_max = 527

    slicing_min = int(win_freq_min/freq[-1]*freq.shape[0])
    slicing_max = int(win_freq_max/freq[-1]*freq.shape[0])

    freq_max = freq[slicing_min+np.argmax(signal_ffted[slicing_min:slicing_max])]
    guess = chr(65+max(0, int(freq_max)-501))
    print(f"[{id}]\tExpected: {expected}, guess: {guess}, freq_max: {freq_max}")

    # plt.plot(freq[slicing_min:slicing_max], signal_ffted[slicing_min:slicing_max])
    # plt.yscale("log")
    # plt.show()
