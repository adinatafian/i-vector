#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import bob.io.base.test_utils
import bob.measure
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(10)

#load hasil skoring
##PM
positif_pm = np.loadtxt('Raw_Pos_PM.txt')
negatif_pm = np.loadtxt('Raw_Neg_PM.txt')
negatif_pm = negatif_pm.reshape(2070,)

z_positif_pm = np.loadtxt('z_pos_pm.txt')
z_positif_pm = z_positif_pm.reshape(46,)
z_negatif_pm = np.loadtxt('z_neg_pm.txt')
z_negatif_pm = z_negatif_pm.reshape(2070,)

t_positif_pm = np.loadtxt('t_pos_pm.txt')
t_positif_pm = t_positif_pm.reshape(46,)
t_negatif_pm = np.loadtxt('t_neg_pm.txt')
t_negatif_pm = t_negatif_pm.reshape(2070,)

zt_positif_pm = np.loadtxt('zt_pos_pm.txt')
zt_positif_pm = zt_positif_pm.reshape(46,)
zt_negatif_pm = np.loadtxt('zt_neg_pm.txt')
zt_negatif_pm = zt_negatif_pm.reshape(2070,)

##PF
positif_pf = np.loadtxt('Raw_Pos_PF.txt')
negatif_pf = np.loadtxt('Raw_Neg_PF.txt')
negatif_pf = negatif_pf.reshape(2652,)

z_positif_pf = np.loadtxt('z_pos_pf.txt')
z_positif_pf = z_positif_pf.reshape(52,)
z_negatif_pf = np.loadtxt('z_neg_pf.txt')
z_negatif_pf = z_negatif_pf.reshape(2652,)

t_positif_pf = np.loadtxt('t_pos_pf.txt')
t_positif_pf = t_positif_pf.reshape(52,)
t_negatif_pf = np.loadtxt('t_neg_pf.txt')
t_negatif_pf = t_negatif_pf.reshape(2652,)

zt_positif_pf = np.loadtxt('zt_pos_pf.txt')
zt_positif_pf = zt_positif_pf.reshape(52,)
zt_negatif_pf = np.loadtxt('zt_neg_pf.txt')
zt_negatif_pf = zt_negatif_pf.reshape(2652,)

##WM
positif_wm = np.loadtxt('Raw_Pos_WM.txt')
negatif_wm = np.loadtxt('Raw_Neg_WM.txt')
negatif_wm = negatif_wm.reshape(2070,)

z_positif_wm = np.loadtxt('z_pos_wm.txt')
z_positif_wm = z_positif_wm.reshape(46,)
z_negatif_wm = np.loadtxt('z_neg_wm.txt')
z_negatif_wm = z_negatif_wm.reshape(2070,)

t_positif_wm = np.loadtxt('t_pos_wm.txt')
t_positif_wm = t_positif_wm.reshape(46,)
t_negatif_wm = np.loadtxt('t_neg_wm.txt')
t_negatif_wm = t_negatif_wm.reshape(2070,)

zt_positif_wm = np.loadtxt('zt_pos_wm.txt')
zt_positif_wm = zt_positif_wm.reshape(46,)
zt_negatif_wm = np.loadtxt('zt_neg_wm.txt')
zt_negatif_wm = zt_negatif_wm.reshape(2070,)

##WF
positif_wf = np.loadtxt('Raw_Pos_WF.txt')
negatif_wf = np.loadtxt('Raw_Neg_WF.txt')
negatif_wf = negatif_wf.reshape(2652,)

z_positif_wf = np.loadtxt('z_pos_wf.txt')
z_positif_wf = z_positif_wf.reshape(52,)
z_negatif_wf = np.loadtxt('z_neg_wf.txt')
z_negatif_wf = z_negatif_wf.reshape(2652,)

t_positif_wf = np.loadtxt('t_pos_wf.txt')
t_positif_wf = t_positif_wf.reshape(52,)
t_negatif_wf = np.loadtxt('t_neg_wf.txt')
t_negatif_wf = t_negatif_wf.reshape(2652,)

zt_positif_wf = np.loadtxt('zt_pos_wf.txt')
zt_positif_wf = zt_positif_wf.reshape(52,)
zt_negatif_wf = np.loadtxt('zt_neg_wf.txt')
zt_negatif_wf = zt_negatif_wf.reshape(2652,)


#Hitung EER,FAR,FRR
##PM
eer_pm, far_pm, frr_pm = bob.measure.eer(negatif_pm, positif_pm, is_sorted=False, also_farfrr=True)
eer_rocch_pm = bob.measure.eer_rocch(negatif_pm, positif_pm)
eer_rocch_zpm = bob.measure.eer_rocch(z_negatif_pm, z_positif_pm)
eer_rocch_tpm = bob.measure.eer_rocch(t_negatif_pm, t_positif_pm)
eer_rocch_ztpm = bob.measure.eer_rocch(zt_negatif_pm, zt_positif_pm)
##PF
eer_pf, far_pf, frr_pf = bob.measure.eer(negatif_pf, positif_pf, is_sorted=False, also_farfrr=True)
eer_rocch_pf = bob.measure.eer_rocch(negatif_pf, positif_pf)
eer_rocch_zpf = bob.measure.eer_rocch(z_negatif_pf, z_positif_pf)
eer_rocch_tpf = bob.measure.eer_rocch(t_negatif_pf, t_positif_pf)
eer_rocch_ztpf = bob.measure.eer_rocch(zt_negatif_pf, zt_positif_pf)
##WM
eer_wm, far_wm, frr_wm = bob.measure.eer(negatif_wm, positif_wm, is_sorted=False, also_farfrr=True)
eer_rocch_wm = bob.measure.eer_rocch(negatif_wm, positif_wm)
eer_rocch_zwm = bob.measure.eer_rocch(z_negatif_wm, z_positif_wm)
eer_rocch_twm = bob.measure.eer_rocch(t_negatif_wm, t_positif_wm)
eer_rocch_ztwm = bob.measure.eer_rocch(zt_negatif_wm, zt_positif_wm)
##WF
eer_wf, far_wf, frr_wf = bob.measure.eer(negatif_wf, positif_wf, is_sorted=False, also_farfrr=True)
eer_rocch_wf = bob.measure.eer_rocch(negatif_wf, positif_wf)
eer_rocch_zwf = bob.measure.eer_rocch(z_negatif_wf, z_positif_wf)
eer_rocch_twf = bob.measure.eer_rocch(t_negatif_wf, t_positif_wf)
eer_rocch_ztwf = bob.measure.eer_rocch(zt_negatif_wf, zt_positif_wf)

#print hasil evaluasi
fileout = open('Evaluasi.txt', 'w')
##PM
print('EER_rocch_pm = %18.10f' % (eer_rocch_pm), file=fileout)
print('far_pm = %18.10f' % (far_pm), file=fileout)
print('frr_pm = %18.10f' % (frr_pm), file=fileout)
print('EER_rocch_zpm = %18.10f' % (eer_rocch_zpm), file=fileout)
print('EER_rocch_tpm = %18.10f' % (eer_rocch_tpm), file=fileout)
print('EER_rocch_ztpm = %18.10f' % (eer_rocch_ztpm), file=fileout)
print('',file=fileout)
##PF
print('EER_rocch_pf = %18.10f' % (eer_rocch_pf), file=fileout)
print('far_pf = %18.10f' % (far_pf), file=fileout)
print('frr_pf = %18.10f' % (frr_pf), file=fileout)
print('EER_rocch_zpf = %18.10f' % (eer_rocch_zpf), file=fileout)
print('EER_rocch_tpf = %18.10f' % (eer_rocch_tpf), file=fileout)
print('EER_rocch_ztpf = %18.10f' % (eer_rocch_ztpf), file=fileout)
print('',file=fileout)
##WM
print('EER_rocch_wm = %18.10f' % (eer_rocch_wm), file=fileout)
print('far_wm = %18.10f' % (far_wm), file=fileout)
print('frr_wm = %18.10f' % (frr_wm), file=fileout)
print('EER_rocch_zwm = %18.10f' % (eer_rocch_zwm), file=fileout)
print('EER_rocch_twm = %18.10f' % (eer_rocch_twm), file=fileout)
print('EER_rocch_ztwm = %18.10f' % (eer_rocch_ztwm), file=fileout)
print('',file=fileout)
##WF
print('EER_rocch_wf = %18.10f' % (eer_rocch_wf), file=fileout)
print('far_wf = %18.10f' % (far_wf), file=fileout)
print('frr_wf = %18.10f' % (frr_wf), file=fileout)
print('EER_rocch_zwf = %18.10f' % (eer_rocch_zwf), file=fileout)
print('EER_rocch_twf = %18.10f' % (eer_rocch_twf), file=fileout)
print('EER_rocch_ztwf = %18.10f' % (eer_rocch_ztwf), file=fileout)
fileout.close()

#plot hasil evaluasi
##plot DET
plt.figure(1)

bob.measure.plot.det(negatif_pm, positif_pm, negatif_pm.shape[0], color=(1,0,0), linestyle='-', label='Percakapan (L)')
bob.measure.plot.det_axis([0.1, 95, 0.1, 95])
plt.plot(bob.measure.ppndf(far_pm), bob.measure.ppndf(frr_pm), 'ro')

bob.measure.plot.det(negatif_pf, positif_pf, negatif_pf.shape[0], color=(0,1,0), linestyle='-', label='Percakapan (P)')
bob.measure.plot.det_axis([0.1, 95, 0.1, 95])
plt.plot(bob.measure.ppndf(far_pf), bob.measure.ppndf(frr_pf), 'go')


bob.measure.plot.det(negatif_wm, positif_wm, negatif_wm.shape[0], color=(0,0,1), linestyle='-', label='Wawancara (L)')
bob.measure.plot.det_axis([0.1, 95, 0.1, 95])
plt.plot(bob.measure.ppndf(far_wm), bob.measure.ppndf(frr_wm), 'bo')


bob.measure.plot.det(negatif_wf, positif_wf, negatif_wf.shape[0], color=(1,1,0), linestyle='-', label='Wawancara (P)')
bob.measure.plot.det_axis([0.1, 95, 0.1, 95])
plt.plot(bob.measure.ppndf(far_wf), bob.measure.ppndf(frr_wf), 'yo')

legend = plt.legend()
plt.grid(True) 
plt.xlabel('FAR (%)')
plt.ylabel('FRR (%)')
plt.title('DET')
plt.savefig('DET.png')

#figure = plt.subplot(2, 1, 1)
#ax = figure.axes
#plt.plot()
#plt.title("Raw scores", fontsize=8)
#plt.hist(negatif, label='Non Target', density=True, stacked=True,
#         color='C1', alpha=0.5, bins=20)

#plt.hist(positif, label='Target', density=True, stacked=True,
#         color='C0', alpha=0.5, bins=8)

#plt.legend(fontsize=8)
#plt.yticks([], [])
#plt.subplot(1,1,1)
plt.figure(2)
plt.subplot(4,1,1)
plt.title('Percakapan (L)')
sns.distplot(negatif_pm.reshape(2070,), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(positif_pm.reshape(46,1), fit=norm, fit_kws={"color":"red"},  kde=False, label='target')
plt.xlim([-0.2, 1.25])
plt.ylabel('Rapat Probabilitas',fontsize=7)
plt.legend(fontsize=8)

plt.subplot(4,1,2)
plt.title('Percakapan (P)')
sns.distplot(negatif_pf.reshape(2652,), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(positif_pf.reshape(52,1), fit=norm, fit_kws={"color":"red"},  kde=False, label='target')
plt.xlim([-0.2, 1.25])
plt.ylabel('Rapat Probabilitas',fontsize=7)
plt.legend(fontsize=8)

plt.subplot(4,1,3)
plt.title('Wawancara (L)')
sns.distplot(negatif_wm.reshape(2070,), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(positif_wm.reshape(46,1), fit=norm, fit_kws={"color":"red"},  kde=False, label='target')
plt.xlim([-0.2, 1.25])
plt.ylabel('Rapat Probabilitas',fontsize=7)
plt.legend(fontsize=8)

plt.subplot(4,1,4)
plt.title('Wawancara (P)')
sns.distplot(negatif_wf.reshape(2652,), fit=norm, fit_kws={"color":"blue"}, kde=False, label='non-target')
sns.distplot(positif_wf.reshape(52,1), fit=norm, fit_kws={"color":"red"},  kde=False, label='target')
plt.xlim([-0.2, 1.25])

plt.xlabel('Cosine Distance', fontsize=9)
plt.ylabel('Rapat Probabilitas',fontsize=7)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig('histogram.png')
