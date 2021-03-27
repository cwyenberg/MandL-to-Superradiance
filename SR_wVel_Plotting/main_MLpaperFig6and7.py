import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pickle

############################################################
############ ML PAPER FIGURE 6 ANNOTATIONS #################

############ BEGIN FIG6B #################

nt = 500
E0 = 1.e-14
I0 = E0*E0

fig, ax1 = plt.subplots()

fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 1.e8)
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(.5, 1.e9)

##### SEMILOG IF FROM z=0 TO z=.6L #####
with open('if_nint30_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
for z_zone in range(0, 3):
    ax1.semilogy(xposns, int_transients[4*z_zone+3, :]/I0, color=color)
##########################

##### SEMILOG TD FROM z=0 TO z=.6L #####
with open('td_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'black'
xposns = np.linspace(0., 1.e8, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color, linestyle='dotted')
plt.text(xcen, 1.3,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 3):
    ax1.semilogy(xposns, int_transients[4*z_zone+3, :]/I0, color=color, linestyle='dotted')
    strfrac = str((4*z_zone+4) / 20)
    plt.text(xcen, 1.3 * int_transients[4*z_zone+3, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

##### Arrow Annotation #####
ax1.annotate("Gibbs phenomena", xy=(1.5e6, 10.), xytext=(5.e7, 40.),
             fontsize=16, fontname='Times New Roman', horizontalalignment='center',
             verticalalignment='center',
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)
ax1.annotate("", xy=(0.985e8, 200.), xytext=(6.35e7, 45.),
             fontsize=16, fontname='Times New Roman', horizontalalignment='center',
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

plt.savefig("fig6b.eps", format='eps')

plt.show()
ax1.cla()

####### BEGIN FIG6A ###############

nt = 500

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.xaxis.set_ticklabels([])
ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.xlim(0., 1.e8)
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(-2.e7, 9.e9)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### LINEAR IF FROM z=.7L TO z=.8L #####
with open('if_nint30_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(13, 16):
    ax1.plot(xposns, int_transients[z_zone, :]/I0, color=color)
##########################

##### LINEAR TD FROM z=.7L TO z=.8L #####
with open('td_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'black'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(13, 16):
    ax1.plot(xposns, int_transients[z_zone, :]/I0, color=color, linestyle='dotted')
    strfrac = str((z_zone+1) / 20)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

############ LEGEND
line = plt.Line2D((6.5e7, 9.5e7),
                  (6.75e9, 6.75e9),
                  lw=1.,
                  color='tab:blue',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 6.75e9,
         'IF algorithm',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

line = plt.Line2D((6.5e7, 9.5e7),
                  (7.875e9, 7.875e9),
                  lw=1.,
                  color='black',
                  linestyle='dotted',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 7.875e9,
         'Time domain reference',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

ax1.annotate("Gibbs phenomena", xy=(2.5e5, 3.75e8), xytext=(5.e7, 2.e8),
             fontsize=16, fontname='Times New Roman', horizontalalignment='center',
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)
ax1.annotate("", xy=(0.99e8, 3.75e8), xytext=(6.4e7, 3.e8),
             fontsize=16, fontname='Times New Roman', horizontalalignment='center',
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

plt.savefig("fig6a.eps", format='eps')

plt.show()
ax1.cla()

####### BEGIN FIG7 ###############

nt = 500

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 5.5)

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.xlim(0., 1.e8)

ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
plt.ylim(-2.e9, 6.5e10)
tx = ax1.yaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

##### LINEAR IF FROM z=.8L TO z=L #####
with open('if_nint30_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'tab:blue'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(0, 3):
    ax1.plot(xposns, int_transients[3*z_zone+13, :]/I0, color=color)
##########################

##### LINEAR TD FROM z=.7L TO z=L #####
with open('td_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

xcen = 5.e7
color = 'black'
xposns = np.linspace(0., 1.e8, nt)
for z_zone in range(0, 3):
    ax1.plot(xposns, int_transients[3*z_zone+13, :]/I0, color=color, linestyle='dotted')
    strfrac = str((3*z_zone+14) / 20)
    plt.text(xcen, int_transients[3*z_zone+13, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

############ LEGEND
line = plt.Line2D((6.5e7, 9.5e7),
                  (4.75e10, 4.75e10),
                  lw=1.,
                  color='tab:blue',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 4.75e10,
         'IF algorithm',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

line = plt.Line2D((6.5e7, 9.5e7),
                  (5.5e10, 5.5e10),
                  lw=1.,
                  color='black',
                  linestyle='dotted',
                  clip_on=False)
ax1.add_line(line)
plt.text(8.e7, 5.5e10,
         'Time domain reference',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom',
         rotation=0,
         clip_on=False)

plt.savefig("fig7.eps", format='eps')

plt.show()
ax1.cla()
