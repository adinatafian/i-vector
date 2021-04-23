#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bob.bio.gmm
import bob.learn.em
import numpy as np
import bob.bio.base

data = np.empty([0,60])

#loading data hasil MFCC
print('loading data . . .')
for i in range (1, 20+1):
	file_ceps = str(i) + '_M_Percakapan_Mic_ceps.txt'
	data_ceps = bob.bio.base.load(file_ceps)
	data = np.vstack((data, data_ceps))
	
print(data.shape)
background_data = np.vsplit(data, data.shape[0])

#parameter i-vector yang digunakan
print('Training i-vector')
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
		#

#pelatihan matriks tv dan ubm kemudian disimpan
ivec.train_projector([background_data], 'projector_file100128.hdf5')
