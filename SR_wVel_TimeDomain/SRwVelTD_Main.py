#################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~ SR_wVel_TimeDomain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ TIME DOMAIN SIMULATION OF THE VELOCITY DEPENDENT MB EQUATIONS ~~~~~~~~~~
#
# LAST UPDATE: 26 March 2021
#
# SUPPLEMENTARY TO THE FOLLOWING PAPER:
# "Generalisation of the Menegozzi & Lamb Maser Algorithm
# to the Transient Superradiance Regime"
# CMW et al.
#
# This code simulates the MB equations across a velocity distribution, intended
# specifically for the transient SR regime.
#
# Simulation parameters required to recover paper figures provided in comments
# and flagged as PAPER_REF: NNN
#
#################################################################################

# Import libraries
import numpy as np
import time                                 # For timing purposes
import pickle                               # For saving out object data to plot in external script

# plotting
import matplotlib as mpl
mpl.use('TkAgg')                            # Fixes a bug on Mac OSX
import matplotlib.pyplot as plt

# Classes
from SRwVelTD_Classes import SimSetup       # Simulation Parameter setup class
from SRwVelTD_Fns import simulate           # simulation function

# Fns
import CorrFns                              # Correlation functions

# ~~~~~~~~~~~~~~~~~~ SIMULATION CONFIG DATA ~~~~~~~~~~~~~~~~~~~~~~
t_dur = 1.e8                                # Simulation time duration (s).  PAPER_REF: 1.e8

# Simulation size parameters:
nt = 500                                    # Number of time points.  PAPER_REF: 500
nz = 401                                    # Number of z posns.  PAPER_REF: 401
                                            #   should be (multiple of n_plt_posns) + 1
nsch = 10                                   # Number of side channels.  PAPER_REF: 10

# Simulation physical characteristics:
length = 2.e15                              # Sample length (cm).  PAPER_REF: 2.e15
wid = 5.4e7                                 # Sample radius (cm).  PAPER_REF: 5.4e7
at_density = 1.5e-12                        # Atomic density (cm^{-3}).  PAPER_REF: 1.5e-12
n0 = 1. * at_density                        # Initial pop inv (cm^{-3}).  PAPER_REF: 1.*at_density
E0_VperM = 1.e-16                           # Incident electric field at z=0 (V/m).  PAPER_REF: 1.e-16

# Medium properties
debye = 3.33564e-28                         # C cm : CONSTANT : One Debye
dip = 0.7 * debye                           # Dipole moment mat elt (C cm).  PAPER_REF: 0.7 * debye
w0 = 2.*np.pi*6.7e9                         # Angular freq of emission (s^-1).  PAPER_REF: 2.*np.pi*6.7e9
t1 = 1.64e7  # 1.64e7                       # Relaxation time constant (s).  PAPER_REF: 1.64e7
t2 = 1.55e6  # 1.55e6                       # Dephasing time constant(s).  PAPER_REF: 1.55e6

# Pump characteristics
gam_n0 = n0 / t1                            # Inversion constant pump term (cm^{-3}/s).
                                            #   PAPER_REF: n0 / t1
gam_p_bloch_en = True                       # Toggle "Bloch angle pumping": to remain consistent with the SR
                                            #   initial tipping angle prescription, enable this feature so that
                                            #   the polarisation pumps along with any inversion pump, and in proportion
                                            #   established by the initial tipping angle prescription;
                                            #   i.e., Lambda^(P) = Lambda^(N) * d * sin(theta_0) at all times.
                                            #   PAPER_REF: True

# Pump pulse parameters
gam_n1 = 0.                                 # Pump pulse amplitude (cm^-3 s^-1).  PAPER_REF: 0.
gam_n_tp = 1.e6                             # Pump pulse duration (s).  PAPER_REF: 1.e6 but irrelevant
gam_n_tau0 = 1.e7                           # Pump pulse delay (s).  PAPER_REF: 1.e7 but irrelevant

fvtype = 'plateau'                          # Velocity distribution profile type (string).  PAPER_REF: 'plateau'

# Random settings
rand_things = False                         # Toggle randomisation of polarisation phases and initial tipping angles
                                            #   PAPER_REF: False
n_rand_runs = 1                             # Number of multiple runs to do for averaging.  PAPER_REF: 1

# Visualisation details:
animate = True                              # Toggle to animate polarisation as simulation executes.
                                            #   PAPER_REF: True but irrelevant
plotstep = 50                               # Number of time steps between visualisation frames of unfolding sim
                                            #   and between progress annunciation.
                                            #   PAPER_REF: 50 but irrelevant
bandstep = 1                                # Visualise every bandstep^th velocity channel. i.e., if 10, only draw
                                            #   every tenth velocity channel when animating unfolding sim.
                                            #   PAPER_REF: 1 but irrelevant
n_plt_posns = 20                            # Number of z-positions higher than z=0.0L to store transients of
                                            #   PAPER_REF: 5 to generate Figure 3, 20 to generate other figures

# ~~~~~~~~~~~~~~~~~~~~~~~ END INPUT DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~

# BEGIN SIMULATION - NO USER ENTRY REQUIRED BEYOND THIS POINT

print("simulating " + str(nt) + " timesteps...")        # Annunciate commencement of simulation
simtime_start = time.time()                             # Start timing the simulation

np.random.seed()    # Seed the random number generator

# Compute some useful quantities
E0 = 100. * E0_VperM                                    # E0 in units of kg cm / (s^2 C) (used by this simulator).

if not rand_things:     # If we're not randomizing things, don't perform multiple runs
    n_rand_runs = 1

# Call the SimSetup class to instantiate an object containing all the useful parameter values,
#   and also initiate some arrays, helpful constants, pump values, and more. See class comments for more details.
sim = SimSetup(nz, nt, nsch,
               t_dur, length, wid, at_density,
               dip, w0, t1, t2, gam_p_bloch_en,
               n0, gam_n0, gam_n1, gam_n_tp, gam_n_tau0,
               fvtype, rand_things, n_plt_posns, E0)

# Validate the stiffness of the simulation
print("The stiffness factor is " + str((sim.dv*sim.nsb)*sim.dt))
if sim.dv*sim.nsb*sim.dt > 1.:
    print("WARNING: EXCESSIVE STIFFNESS")

# Allocate the totalising (averaging) arrays
e_field_trans_avg = np.zeros((sim.n_plt_posns, sim.nt), dtype=complex)
int_trans_avg = np.zeros((sim.n_plt_posns, sim.nt), dtype=float)
p_trans_avg = np.zeros((sim.n_plt_posns, sim.nch, sim.nt), dtype=complex)
inv_trans_avg = np.zeros((sim.n_plt_posns, sim.nch, sim.nt), dtype=float)

run_scalar = 1.0/float(n_rand_runs)         # Averaging factor

# Simulate the system the requested number of times:
for run in range(0, n_rand_runs):
    print("Performing run " + str(run+1) + " of " + str(n_rand_runs) + "...")
    (int_tot_transients,
     e_field_transients,
     p_transients,
     inv_transients,
     pump_transient) = simulate(sim, animate, plotstep, bandstep)
    e_field_trans_avg += run_scalar * e_field_transients
    int_trans_avg += run_scalar * np.real(np.multiply(e_field_transients, np.conjugate(e_field_transients)))
    p_trans_avg += run_scalar * p_transients
    inv_trans_avg += run_scalar * inv_transients

# Pickle the data for plotting outside of this python script
with open('td_intensities.pickle', 'wb') as f:
    pickle.dump(int_trans_avg, f)

with open('td_inversions.pickle', 'wb') as f:
    pickle.dump(inv_trans_avg, f)

with open('td_polarizations.pickle', 'wb') as f:
    pickle.dump(p_trans_avg, f)

int_from_e_trans_avg = np.real(np.multiply(np.conjugate(e_field_trans_avg), e_field_trans_avg))
total_energy = sim.dt * np.sum(int_trans_avg[sim.n_plt_posns-1, :])
energy_string = "Total energy (arb units, see code) {:.2e}".format(total_energy)

# Peak intensity value (arb units)  and time of occurrence
peak_int = np.amax(int_trans_avg[sim.n_plt_posns-1, :])
peak_time = float(np.argmax(int_trans_avg[sim.n_plt_posns-1, :])) * sim.dt
intensity_string = "Peak intensity (arb units, see code) of {:.2e}".format(peak_int) +\
                   " @ t = {:.2e}".format(peak_time)
print(intensity_string)
print(energy_string)

# Next, the width. We use a mean square deviation metric:
tvec = np.linspace(0, t_dur, num=nt, endpoint=True)
int_sum = np.sum(int_trans_avg[sim.n_plt_posns-1, :])
mean_t = np.dot(int_trans_avg[sim.n_plt_posns-1, :], tvec) / int_sum
dev_vec = tvec - mean_t
del_t = (1.0/np.sqrt(int_sum)) * np.sqrt(np.dot(int_trans_avg[sim.n_plt_posns-1, :], np.multiply(dev_vec, dev_vec)))
print("Pulse width of {:.2e}".format(del_t) + " s.")

# Save the spectrum:
x = np.linspace(0, t_dur, num=nt, endpoint=True)
plt.cla()
p_spectra = np.empty((sim.nch, sim.nt), dtype=float)
for k in range(0, sim.nsb+1):
    p_spectra[k+sim.nsb, :] = CorrFns.power_spectrum(p_trans_avg[-1, k, :])
for k in range(sim.nsb+1, sim.nch):
    p_spectra[k-sim.nsb-1, :] = CorrFns.power_spectrum(p_trans_avg[-1, k, :])
plt.imshow(np.transpose(p_spectra))
plt.savefig("SpectralRaster", dpi=160)
plt.cla()

simtime_end = time.time()

print("Total execution time " + str(simtime_end-simtime_start) + " s.")
