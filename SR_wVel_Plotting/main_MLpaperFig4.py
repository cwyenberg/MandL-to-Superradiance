import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

############################################################
############ ML PAPER FIGURE 4 ANNOTATIONS #################

nt = 500
E0 = 1.e-14             # In cm length units
I0 = E0*E0
fig, ax1 = plt.subplots()

fig.set_size_inches(8, 10)
plt.xlim(0., 10.)
plt.ylim(.5, 1.e13)

ax1.set_xlabel('Varying LMI Fidelity', fontsize=16, fontname='Times New Roman')
ax1.axes.xaxis.set_visible(False)
ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.tick_params(axis="x",
                which="both",
                bottom=False,
                top=False)
plt.yticks(fontsize=16, fontname='Times New Roman')

##### NINT 10 LEVELS #####
with open('ml_nint10_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)
xcen = 1.

color = 'tab:blue'
xposns = np.linspace(xcen-.75, xcen+.75, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
plt.text(xcen, 1.,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 5):
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0, color=color)
    strfrac = str((z_zone+1) / 5)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

##### NINT 8 LEVELS #####
with open('ml_nint8_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)
xcen = 3.

color = 'tab:blue'
xposns = np.linspace(xcen-.75, xcen+.75, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
plt.text(xcen, 1.,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 5):
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0, color=color)
    strfrac = str((z_zone+1) / 5)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

##### NINT 6 LEVELS #####
with open('ml_nint6_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)
xcen = 5.

color = 'tab:blue'
xposns = np.linspace(xcen-.75, xcen+.75, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
plt.text(xcen, 1.,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 5):
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0, color=color)
    strfrac = str((z_zone+1) / 5)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

##### NINT 4 LEVELS #####
with open('ml_nint4_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)
xcen = 7.

color = 'tab:blue'
xposns = np.linspace(xcen-.75, xcen+.75, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
plt.text(xcen, 1.,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 5):
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0, color=color)
    strfrac = str((z_zone+1) / 5)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

##### NINT 2 LEVELS #####
with open('ml_nint2_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)
xcen = 9.

color = 'tab:blue'
xposns = np.linspace(xcen-.75, xcen+.75, nt)
ax1.semilogy(xposns, np.full(nt, 1.), color=color)
plt.text(xcen, 1.,
         "z = 0.0 L",
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')
for z_zone in range(0, 5):
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0, color=color)
    strfrac = str((z_zone+1) / 5)
    plt.text(xcen, int_transients[z_zone, int(nt/2)]/I0,
             "z = " + strfrac + " L",
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='bottom')
##########################

# ANNOTATIONS AND MARKUPS:

# Vertical partitions
for xind in range(0, 4):
    line = plt.Line2D((2.+2.*xind, 2.+2.*xind),
                      (.15, 1.e13),
                      lw=1.,
                      color='tab:gray',
                      linestyle='dashed',
                      clip_on=False)
    ax1.add_line(line)

# N_int Labels
for xind in range(0,5):
    plt.text(2*xind+1, .2,
             r'$N_{\mathrm{int}}$ = ' + str(10-2*xind),
             fontsize=16,
             fontname='Times New Roman',
             horizontalalignment='center',
             verticalalignment='center')

# Fidelity progression arrow
ax1.annotate("", xy=(10., 2.e13), xytext=(0., 2.e13),
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)
# Fidelity label
plt.text(5., 2.5e13,
         'decreasing LMI fidelity',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='bottom')

# Saturation progression arrow
ax1.annotate("", xy=(10.25, 5.e12), xytext=(10.25, 1.),
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)
# Fidelity label
plt.text(10.5, 5.e5,
         'toward saturated maser / SR domain',
         fontsize=16,
         fontname='Times New Roman',
         horizontalalignment='center',
         verticalalignment='center',
         rotation=90,
         clip_on=False)

plt.savefig("fig4.eps", format='eps')
ax1.cla()
