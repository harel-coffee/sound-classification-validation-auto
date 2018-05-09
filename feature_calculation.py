#!/usr/bin/env python
# -*- coding: utf-8 -*-

import soundfile as sf
import glob
import xtract
import csv

# parameters for libxtract
win_size = 8192
sub_win_divider = 16
num_lpc = 10
num_filters_mfcc = 10
to_process = xtract.doubleArray(win_size)
# where are the files
files = glob.glob('./Dataset_mono_16k/*.wav')
features = []
classes = []
file_count = 0
# go through all files
for audio_file in files:
    file_count +=1
    print('File no. %s, %.2f %%' % (file_count, 100.0*file_count/len(files)))
    # find the class from the name
    audio_class = (audio_file.split('__')[1]).split('_')[0]
    # read the file
    data, rate = sf.read(audio_file)
    start = 0
    # go through chunks of size win_size
    while start < len(data):
        chunk = data[start:min(start+win_size, len(data))]
        if len(chunk) == win_size:
            for i in range(win_size):
                to_process[i] = chunk[i]

            # calculate ZCR
            zcr = xtract.xtract_zcr(to_process, win_size, None)[1]

            # calculate HZCRR
            start_sub_window = 0
            count = 0
            while start_sub_window < win_size:
                sub_chunk = chunk[start_sub_window:min((start_sub_window+win_size/sub_win_divider), win_size)]
                sub_process = xtract.doubleArray(len(sub_chunk))
                for i in range(len(sub_chunk)):
                    sub_process[i] = sub_chunk[i]
                sub_zcr = xtract.xtract_zcr(sub_process, len(sub_chunk), None)[1]
                if sub_zcr > 1.5 * zcr:
                    count += 1
                start_sub_window += win_size/sub_win_divider

            hzcrr = 1.0 * count / (2.0 * win_size/sub_win_divider)
            # calculate mean
            mean = xtract.xtract_mean(to_process, win_size, None)[1]
            argv = xtract.doubleArray(1)
            argv[0] = mean
            # calculate variance
            variance = xtract.xtract_variance(to_process, win_size, argv)[1]
            # calculate std dev
            argv[0] = variance
            std_dev = xtract.xtract_standard_deviation(to_process, win_size, argv)[1]
            # calculate kurtosis
            argv = xtract.doubleArray(2)
            argv[0] = mean
            argv[1] = std_dev
            kurtosis = xtract.xtract_kurtosis(to_process, win_size, argv)[1]
            # calculate skewness
            skewness = xtract.xtract_skewness(to_process, win_size, argv)[1]
            # do fft
            argv = xtract.doubleArray(4)
            argv[0] = 16000.0/win_size
            argv[1] = xtract.XTRACT_MAGNITUDE_SPECTRUM
            argv[2] = 0
            argv[3] = 0
            spectrum = xtract.doubleArray(win_size)
            xtract.xtract_init_fft(win_size, xtract.XTRACT_SPECTRUM)
            xtract.xtract_spectrum(to_process, win_size, argv, spectrum)
            xtract.xtract_free_fft()
	    # calculate spectral mean
            spectral_mean = xtract.xtract_spectral_mean(spectrum, win_size, None)[1]
            # calculate spectral variance
            argv = xtract.doubleArray(1)
            argv[0] = spectral_mean
            spectral_variance = xtract.xtract_spectral_variance(spectrum, win_size, argv)[1]
            # calculate spectral std dev
            argv[0] = spectral_variance
            spectral_std_dev = xtract.xtract_spectral_standard_deviation(spectrum, win_size, argv)[1]
            # calculate spectral centroid
            spectral_centroid = xtract.xtract_spectral_centroid(spectrum, win_size, None)[1]
            # calculate spectral kurtosis
            argv = xtract.doubleArray(2)
            argv[0] = spectral_mean
            argv[1] = spectral_std_dev
            spectral_kurtosis = xtract.xtract_spectral_kurtosis(spectrum, win_size, argv)[1]
            # calculate spectral skewness
            spectral_skewness = xtract.xtract_spectral_skewness(spectrum, win_size, argv)[1]
            # calculate sharpness
            argv = xtract.doubleArray(win_size/2)
            for i in range(win_size/2):
                argv[i] = spectrum[1]
            sharpness = xtract.xtract_sharpness(argv, win_size/2, None)[1]
            # calculate bark coefficients
            bark_limits = xtract.intArray(24)
            xtract.xtract_init_bark(win_size/2, 16000, bark_limits)
            bark = xtract.doubleArray(24)
            xtract.xtract_bark_coefficients(argv, win_size/2, bark_limits, bark)
            # calculate loudness
            loudness = xtract.xtract_loudness(bark, 24, None)[1]
            # calculate rms
            rms = xtract.xtract_rms_amplitude(to_process, win_size, None)[1]
            # calculate auto correlation
            auto_correlation = xtract.doubleArray(win_size+1)
            xtract.xtract_autocorrelation(to_process, win_size, None, auto_correlation)
            # calculate lpc
            reflection_lpc = xtract.doubleArray(2*num_lpc)
            xtract.xtract_lpc(auto_correlation, num_lpc+1, None, reflection_lpc)
            lpc = [0.]*num_lpc
            for i in range(num_lpc):
                lpc[i] = reflection_lpc[num_lpc+i]
            # calculate mfcc
            mfcc_bank = xtract.create_filterbank(num_filters_mfcc, win_size)
            xtract.xtract_init_mfcc(win_size/2, 8000, xtract.XTRACT_EQUAL_GAIN, 20, 8000,
                                    mfcc_bank.n_filters, mfcc_bank.filters)
            mfcc = xtract.doubleArray(24)
            xtract.xtract_mfcc(spectrum, win_size/2, mfcc_bank, mfcc)
            # compose feature vector
            feature_vector = [audio_class, zcr, hzcrr, kurtosis, skewness, spectral_mean, spectral_variance, spectral_std_dev,
                              spectral_centroid, spectral_kurtosis, spectral_skewness, loudness, rms]
            for i in range(len(lpc)):
                feature_vector.append(lpc[i])
 
            for i in range(num_filters_mfcc):
                feature_vector.append(mfcc[i])
            for i in range(24):
                feature_vector.append(bark[i])
            # add feature vector and class to the data
            features.append(feature_vector)
            classes.append(audio_class)

        start = start+win_size

# save features
with open("features_8192_512.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(features)
