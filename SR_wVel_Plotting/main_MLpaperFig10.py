# GENERATE FIGURE 10 FROM DATA

import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('font', family='Times New Roman')
import matplotlib.pyplot as plt
import numpy as np
import pickle

####### BEGIN FIG10A ###############

nt = 500
E0 = 1.e-14
I0 = E0*E0

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 5.e7)

ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(0., 6.e7)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### Open the files #####
with open('if_nint30_intensities.pickle', 'rb') as f:
    int_transients_nint30 = pickle.load(f)
with open('if_nint20_intensities.pickle', 'rb') as f:
    int_transients_nint20 = pickle.load(f)
with open('if_nint10_intensities.pickle', 'rb') as f:
    int_transients_nint10 = pickle.load(f)
with open('if_nint5_intensities.pickle', 'rb') as f:
    int_transients_nint5 = pickle.load(f)

color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
ax1.plot(xposns, int_transients_nint30[11, :]/I0, color=color)
ax1.plot(xposns, int_transients_nint20[11, :]/I0, color=color, linestyle='dashed')
ax1.plot(xposns, int_transients_nint10[11, :]/I0, color=color, linestyle='dashdot')
ax1.plot(xposns, int_transients_nint5[11, :]/I0, color=color, linestyle='dotted')

ax1.legend([r'N$_\mathrm{int}=30$',
            r'N$_\mathrm{int}=20$',
            r'N$_\mathrm{int}=10$',
            r'N$_\mathrm{int}=5$'],
           fontsize=16)

plt.text(4.e7, 3.e7,
                 r'z = $0.6$ L ',
                 fontsize=16,
                 fontname='Times New Roman',
                 horizontalalignment='center',
                 verticalalignment='bottom')

plt.savefig("fig10a.eps", format='eps')

plt.show()

ax1.cla()


####### BEGIN FIG10B ###############

nt = 500

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 5.e7)

ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(0., 8.e10)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### Open the files #####
with open('if_nint30_intensities.pickle', 'rb') as f:
    int_transients_nint30 = pickle.load(f)
with open('if_nint20_intensities.pickle', 'rb') as f:
    int_transients_nint20 = pickle.load(f)
with open('if_nint10_intensities.pickle', 'rb') as f:
    int_transients_nint10 = pickle.load(f)
with open('if_nint5_intensities.pickle', 'rb') as f:
    int_transients_nint5 = pickle.load(f)

color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
ax1.plot(xposns, int_transients_nint30[19, :]/I0, color=color)
ax1.plot(xposns, int_transients_nint20[19, :]/I0, color=color, linestyle='dashed')
ax1.plot(xposns, int_transients_nint10[19, :]/I0, color=color, linestyle='dashdot')
ax1.plot(xposns, int_transients_nint5[19, :]/I0, color=color, linestyle='dotted')

ax1.legend([r'N$_\mathrm{int}=30$',
            r'N$_\mathrm{int}=20$',
            r'N$_\mathrm{int}=10$',
            r'N$_\mathrm{int}=5$'],
           fontsize=16)

plt.text(4.e7, 7.e10,
                 r'z = L',
                 fontsize=16,
                 fontname='Times New Roman',
                 horizontalalignment='center',
                 verticalalignment='bottom')

plt.savefig("fig10b.eps", format='eps')

plt.show()

ax1.cla()
