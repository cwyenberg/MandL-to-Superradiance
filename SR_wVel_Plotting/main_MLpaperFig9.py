import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pickle

####### BEGIN FIG9 ###############

deb_fac = 3.33564e-28   # Coulomb cm

nt = 500

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 1.e8)

ax1.set_ylabel(r'Im(P$_{p=0})$ (D cm$^{-3}$)', fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(-2.5e-41 / deb_fac, 11.e-41 / deb_fac)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### LINEAR IF FROM z=.6L TO z=L #####
with open('if_nint30_polarizations.pickle', 'rb') as f:
    pol_transients = pickle.load(f)

xcen = 5.e7
color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(2, 5):
    ax1.plot(xposns, np.imag(pol_transients[4*z_zone+3, :])/deb_fac, color=color)
##########################

##### LINEAR TD FROM z=.6L TO z=L #####
with open('td_polarizations.pickle', 'rb') as f:
    pol_transients = pickle.load(f)

xcen = 5.e7
color = 'black'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(2, 5):
    ax1.plot(xposns, (1./deb_fac)*np.imag(pol_transients[4*z_zone+3, 0, :]), color=color, linestyle='dotted')
##########################

############ LEGEND
line = plt.Line2D((6.5e7, 9.5e7),
                  (8.e-41/deb_fac, 8.e-41/deb_fac),
                  lw=1.,
                  color='tab:blue',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 8.e-41/deb_fac,
         'IF algorithm',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

line = plt.Line2D((6.5e7, 9.5e7),
                  (9.e-41/deb_fac, 9.e-41/deb_fac),
                  lw=1.,
                  color='black',
                  linestyle='dotted',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 9.e-41/deb_fac,
         'Time domain reference',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

# Name the transients
plt.text(1.e7, 9.e-41/deb_fac,
         'z = L',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='left',
         verticalalignment='center',
         rotation=0,
         clip_on=False)
plt.text(1.6e7, 6.e-41/deb_fac,
         'z = 0.8 L',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='left',
         verticalalignment='center',
         rotation=0,
         clip_on=False)
plt.text(5.e7, 0.2e-41/deb_fac,
         'z = 0.6 L',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         clip_on=False)

plt.savefig("fig9.eps", format='eps')

plt.show()
ax1.cla()
