import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

################################################################
############ ML PAPER FIGURE 3 AND ANNOTATIONS #################

nt = 500
t_dur = 1.e8
xposns = np.linspace(0., 1.e8, nt)
n_plt_posns = 5

fig, ax1 = plt.subplots()
fig.set_size_inches(8, 8)

with open('td_intensities.pickle', 'rb') as f:
    int_transients = pickle.load(f)

I0 = int_transients[0, 0]

# Plot the intensity transients (and pump retained on plot)

# At z=0.0L, simply plot unity
color = 'tab:blue'
ax1.semilogy(xposns, np.full(nt, 1., dtype=float),
             linewidth=1.,
             #linestyle='dotted',
             color=color)

plt.text(.8 * t_dur, .9,
             "z = 0.0 L", fontsize=16, fontname='Times New Roman',
             verticalalignment='bottom')

# Plot for multiple positions beyond z=0.0L
for z_zone in range(0, n_plt_posns):
    strFrac = str((z_zone+1) / n_plt_posns)
    if strFrac == '1 ':
        strFrac = ''
    ax1.semilogy(xposns, int_transients[z_zone, :]/I0,
                 linewidth=1.,
                 #linestyle='dotted',
                 color=color)
    plt.text(.8 * t_dur, .9 * int_transients[z_zone, int(.8*nt)]/I0,
             "z = " + strFrac + " L", fontsize=16, fontname='Times New Roman',
             verticalalignment='bottom')

ax1.set_xlabel(r'τ (s)', fontsize=16, fontname='Times New Roman')
ax1.set_ylabel(r'I / I$_0$', fontsize=16, fontname='Times New Roman')
plt.xticks(fontsize=16, fontname='Times New Roman')
tx = ax1.xaxis.get_offset_text()
tx.set_fontsize(16)
tx.set_fontname('Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')

plt.xlim(0., t_dur)
plt.ylim(.5, 1.e11)

# Draw a Rectangle patch around the SR transient
rect = mpl.patches.Rectangle((0, 5.e8), 5.e7, 9.e10,
                             linewidth=1.5,
                             linestyle='dashed',
                             edgecolor='tab:gray',
                             facecolor='none')

# Add the patch to the Axes
ax1.add_patch(rect)

# Annotate
plt.text(3.e7, 3.e10,
         "SR transient",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='bottom')

# Demarcate regimes

# Saturated regime
plt.text(1.075e8, 1.e9,
         "saturated",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center',
         rotation=270)
plt.text(1.05e8, 1.e9,
         "maser",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center',
         rotation=270)
line = plt.Line2D((1.01e8, 1.05e8), (1.6e10, 1.6e10), lw=1.5, color='tab:gray', clip_on=False)
ax1.add_line(line)
line = plt.Line2D((1.01e8, 1.05e8), (6.e7, 6.e7), lw=1.5, linestyle='dotted', color='tab:gray', clip_on=False)
ax1.add_line(line)
ax1.annotate("", xy=(1.02e8, 1.52e10), xytext=(1.02e8, 6.3e7),
             arrowprops=dict(arrowstyle='<->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

# Unsaturated regime
plt.text(1.075e8, 1.e4,
         "unsaturated",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center',
         rotation=270)
plt.text(1.05e8, 1.e4,
         "maser",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center',
         rotation=270)
line = plt.Line2D((1.01e8, 1.05e8), (1.e0, 1.e0), lw=1.5, color='tab:gray', clip_on=False)
ax1.add_line(line)
ax1.annotate("", xy=(1.02e8, 5.7e7), xytext=(1.02e8, 1.05e0),
             arrowprops=dict(arrowstyle='<->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

# Transient domain
plt.text(2.5e7, 8.,
         "transient",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center')

line = plt.Line2D((5.e7, 5.e7), (.5, 10.), lw=1.5, linestyle='dotted', color='tab:gray', clip_on=False)
ax1.add_line(line)
ax1.annotate("", xy=(0., 5.), xytext=(5.e7, 5.),
             arrowprops=dict(arrowstyle='<->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

# Steady state domain
plt.text(7.5e7, 8.,
         "steady state",
         fontsize=16, fontname='Times New Roman',
         verticalalignment='center',
         horizontalalignment='center')

ax1.annotate("", xy=(1.e8, 5.), xytext=(5.05e7, 5.),
             arrowprops=dict(arrowstyle='->', color='tab:gray', mutation_scale=16.),
             annotation_clip=False)

plt.savefig("testout.eps", format='eps')
ax1.cla()

############################################################
