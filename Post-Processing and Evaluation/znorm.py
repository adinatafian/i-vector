#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:30:15 2018

@author: speakerrec
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import bob.learn.linear
import bob.learn.em
import bob.io.base
import bob.measure

#openfile
skor_non = np.loadtxt('Raw_Neg_PF.txt') # non-target
skor_tar = np.loadtxt('Raw_Pos_PF.txt') # target

z_non_score = np.loadtxt('z_neg_pf.txt') # non-target
z_tar_score = np.loadtxt('z_pos_pf.txt') # target

plt.figure(1)
plt.subplot(4, 1, 1)
plt.title("Raw Percakapan Perempuan", fontsize=10)
sns.distplot(skor_non.reshape(2652,1), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(skor_tar.reshape(52,1), fit=norm, fit_kws={"color":"red"},  kde=False, label='target')
plt.xlabel('Cosine Distance', fontsize=9)
plt.ylabel('Rapat Frekuensi',fontsize=7)
plt.legend(fontsize=8)

plt.subplot(4, 1, 2)
plt.title("Z-Norm Percakapan Perempuan", fontsize=10)
sns.distplot(z_non_score.reshape(2652,1), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(z_tar_score.reshape(52,1), fit=norm, fit_kws={"color":"red"}, kde=False, label='target')
plt.xlabel('Skor', fontsize=9)
plt.ylabel('Rapat Frekuensi', fontsize=7)
plt.legend(fontsize=8)




t_non_score = np.loadtxt('t_neg_pf.txt') # non-target
t_tar_score = np.loadtxt('t_pos_pf.txt') # target

plt.subplot(4, 1, 3)
plt.title("T-Norm Percakapan Perempuan", fontsize=10)
sns.distplot(t_non_score.reshape(2652,1), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(t_tar_score.reshape(52,1), fit=norm, fit_kws={"color":"red"}, kde=False, label='target')
plt.xlabel('Skor', fontsize=9)
plt.ylabel('Rapat Frekuensi', fontsize=7)
plt.legend(fontsize=8)

#plt.tight_layout()
#plt.savefig('tnorm_wf.png')


zt_non_score = np.loadtxt('zt_neg_pf.txt') # non-target
zt_tar_score = np.loadtxt('zt_pos_pf.txt') # target
#plt.figure(3)
#plt.subplot(2, 1, 1)
#plt.title("Sebelum ZT-norm", fontsize=10)
#sns.distplot(skor_non.reshape(2652,1), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
#sns.distplot(skor_tar.reshape(52,1), fit=norm, fit_kws=	   {"color":"red"},  kde=False, label='target')
#plt.xlabel('Cosine Distance', fontsize=9)
#plt.ylabel('Probabilitas',fontsize=9)
#plt.legend(fontsize=8)

plt.subplot(4, 1, 4)
plt.title("ZT-Norm Percakapan Perempuan", fontsize=10)
sns.distplot(zt_non_score.reshape(2652,1), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(zt_tar_score.reshape(52,1), fit=norm, fit_kws={"color":"red"}, kde=False, label='target')
plt.xlabel('Skor', fontsize=9)
plt.ylabel('Rapat Frekuensi', fontsize=7)
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('norm_pf.png')