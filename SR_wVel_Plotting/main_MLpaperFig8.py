import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pickle

####### BEGIN FIG8 ###############

nt = 500

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 1.e8)

ax1.set_ylabel(r'N$_{p=0}$ (cm$^{-3}$)', fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(-1.5*1.75e-13, 1.5*5.75e-13)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### LINEAR IF FROM z=.6L TO z=L #####
with open('if_nint30_inversions.pickle', 'rb') as f:
    inv_transients = pickle.load(f)

xcen = 5.e7
color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(6, 10):
    ax1.plot(xposns, inv_transients[2*z_zone+1, :], color=color)
##########################

##### LINEAR TD FROM z=.6L TO z=L #####
with open('td_inversions.pickle', 'rb') as f:
    inv_transients = pickle.load(f)

xcen = 5.e7
color = 'black'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(6, 10):
    ax1.plot(xposns, inv_transients[2*z_zone+1, 0, :], color=color, linestyle='dotted')
    strfrac = str((2*z_zone+2) / 20)
    if z_zone<8:
        plt.text(xcen, inv_transients[2*z_zone+1, 0, int(nt/2)],
                 "z = " + strfrac + " L",
                 fontsize=16,
                 fontname='Times New Roman',
                 horizontalalignment='center',
                 verticalalignment='bottom')
##########################

############ LEGEND
line = plt.Line2D((6.5e7, 9.5e7),
                  (1.5*4.25e-13, 1.5*4.25e-13),
                  lw=1.,
                  color='tab:blue',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 1.5*4.25e-13,
         'IF algorithm',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

line = plt.Line2D((6.5e7, 9.5e7),
                  (1.5*5.e-13, 1.5*5.e-13),
                  lw=1.,
                  color='black',
                  linestyle='dotted',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 1.5*5.e-13,
         'Time domain reference',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

# Name the bottom two transients
plt.text(3.e7, -1.5*1.e-13,
         'z = L',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         clip_on=False)
line = plt.Line2D((3.e7, 3.e7),
                  (-1.5*.7e-13, 1.5*.6e-13),
                  lw=1.,
                  linestyle='dashed',
                  color='tab:gray',
                  clip_on=False)
ax1.add_line(line)

plt.text(7.e7, -1.5*1.e-13,
         'z = 0.9 L',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=0,
         clip_on=False)
line = plt.Line2D((7.e7, 7.e7),
                  (-1.5*.7e-13, 1.5*.85e-13),
                  lw=1.,
                  linestyle='dashed',
                  color='tab:gray',
                  clip_on=False)
ax1.add_line(line)

plt.savefig("fig8.eps", format='eps')

plt.show()
ax1.cla()
