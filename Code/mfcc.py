#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bob.bio.base
import bob.bio.spear
import numpy as np
import scipy.io.wavfile

#parameter voice activity detection
vad_proc = bob.bio.spear.preprocessor.Energy_2Gauss(win_length_ms=20,
                                                     win_shift_ms=10)
                                                    
#parameter MFCC
cep_extr = bob.bio.spear.extractor.Cepstral(win_length_ms=20,
                                            win_shift_ms=10,
                                            n_filters=24,
                                            f_min=200,
                                            f_max=3800,
                                            with_energy=True,
                                            with_delta=True,
                                            with_delta_delta=True,
                                            n_ceps=19,
                                            pre_emphasis_coef=0.97,
                                            normalize_flag=True,
                                            features_mask=np.arange(0,60))

for i in range (1, 46+1):
    file_in = str(i) + '_M_Percakapan_Mic.wav'
    print(file_in)
    rate, audio = scipy.io.wavfile.read(file_in)
    data = np.cast['float'](audio)
    #ekstraksi vad
    rate, signal, vad_labels = vad_proc((rate, data))
    #ekstraksi MFCC
    ceps = cep_extr((rate, signal, vad_labels))
    #penyimpanan hasil ekstraksi
    np.set_printoptions(threshold=np.Infinity)
    file_ceps = str(i) + '_M_Percakapan_Mic_ceps.txt'
    bob.io.base.save(ceps, file_ceps)
