#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bob.bio.gmm
import bob.learn.em
import numpy as np
import bob.bio.base

#parameter i-vector
ivec = bob.bio.gmm.algorithm.IVector(
		#i-vector parameter
		subspace_dimension_of_t=100,
		tv_training_iterations=25,
		update_sigma=True,
		use_whitening=True,
		#gmm parameter
		number_of_gaussians=128,
		use_lda=False,
		use_wccn=False,
		use_plda=False,
		lda_dim=None,
		lda_strip_to_rank=True,
		plda_dim_F=50,
		plda_dim_G=50,
		plda_training_iterations=100,
		)

#loading matriks tv dan ubm hasil latihan
ivec.load_projector('projector_file100128.hdf5')


for i in range (1, 46+1):
    #penyimpanan skor
    fileout = open('LOG_run_ivec_' + str(i) + '.txt', 'w')

    for iprobe in range (1, 46+1):
        #pembagian data untuk sampel K
        f_read_model = bob.bio.base.load(str(i) + '_M_Percakapan_Mic_ceps.txt')
        Nhalf = int(f_read_model.shape[0]/2)
        f_model = f_read_model[0:Nhalf,:]

        #pembagian data untuk sampel UK
        f_read_probe = bob.bio.base.load(str(iprobe) + '_M_Percakapan_Mic_ceps.txt')
        Nhalf = int(f_read_probe.shape[0]/2)
        f_probe = f_read_probe[Nhalf:,:]

        print('Client = %4d, probe = %4d' % (i, iprobe), file=fileout)
        print('Client = %4d, probe = %4d' % (i, iprobe))

        #skoring K dan UK dengan cosine distance
        iv_model = ivec.project(f_model)
        iv_probe = ivec.project(f_probe)
        score = ivec.score(iv_model, iv_probe)
        print(score, file=fileout)
        print('Score %18.10f' % score )

    fileout.close()
