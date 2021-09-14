#################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~ SR_wVel_TimeDomain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ TIME DOMAIN SIMULATION OF THE VELOCITY DEPENDENT MB EQUATIONS ~~~~~~~~~~
#
# LAST UPDATE: 9 September 2021
#
# SUPPLEMENTARY TO THE FOLLOWING PAPER:
# "Generalisation of the Menegozzi & Lamb Maser Algorithm
# to the Transient Superradiance Regime"
# CMW et al.
#
# This code simulates the MB equations across a velocity distribution, intended
# specifically for the transient SR regime.
#
# Simulation parameters required to recover paper figures
# provided in parameter file `params_MLPaper.py'
#
# CHANGELOG:
#
# 9 September 2021
#  Migrated all parameters to a parameter file system and cleaned up the setup
#  code appropriately. Introduced multiple parameter files, including one to
#  recreate simulations from the triggered FRB paper,
# 'Triggered superradiance and fast radio bursts',
#  https://doi.org/10.1093/mnras/sty3046
#
#################################################################################

# Import libraries
import numpy as np
import pickle                               # For saving out object data to plot in external script

# Plotting

# Classes
from SRwVelTD_Classes import SimSetup       # Simulation Parameter setup class
from SRwVelTD_Fns import *

# Fns
from CorrFns import *                            # Correlation functions

# ~~~~~~~~~~~~~~~~~~ IMPORT CONFIGURATION PARAMETERS ~~~~~~~~~~~~~~~~~~~~
import params_cohtrigpulse as params                # Change the parameter filename to suit simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ BEGIN SIMULATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("simulating " + str(params.nt) + " timesteps...")        # Annunciate commencement of simulation
simtime_start = time.time()                             # Start timing the simulation

np.random.seed()    # Seed the random number generator

# Call the SimSetup class to instantiate an object containing all the useful parameter values,
#   and also initiate some arrays, helpful constants, pump values, and more. See class comments for more details.
sim = SimSetup(params)

# Validate the stiffness of the simulation
if (params.fvtype == 'plateau') or (params.fvtype == 'gaussian'):
    stiffness = sim.domega * sim.nsch * sim.dt
else:
    stiffness = sim.domega * sim.dt * (sim.nsch + sim.v_sep)
print("Using a velocity distribution of type: " + params.fvtype + ",")
print("for which the stiffness factor is " + str(stiffness) + ".")
if stiffness > 1.:
    print("WARNING: EXCESSIVE STIFFNESS")

# Allocate the totalising (averaging) arrays
e_field_trans_avg = np.zeros((sim.n_plt_posns, sim.nt_frac), dtype=complex)
int_trans_avg = np.zeros((sim.n_plt_posns, sim.nt_frac), dtype=float)
p_trans_avg = np.zeros((sim.n_plt_posns, sim.nch, sim.nt_frac), dtype=complex)
inv_trans_avg = np.zeros((sim.n_plt_posns, sim.nch, sim.nt_frac), dtype=float)

run_scalar = 1. / float(params.n_rand_runs)         # Averaging factor

# Simulate the system the requested number of times:
for run in range(0, sim.n_rand_runs):
    print("Performing run " + str(run+1) + " of " + str(params.n_rand_runs) + "...")
    (int_tot_transients,
     e_field_transients,
     p_transients,
     inv_transients,
     pump_transient) = simulate(sim)
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
tvec = np.linspace(0, sim.t_dur_frac, num=sim.nt_frac, endpoint=True)
int_sum = np.sum(int_trans_avg[sim.n_plt_posns-1, :])
mean_t = np.dot(int_trans_avg[sim.n_plt_posns-1, :], tvec) / int_sum
dev_vec = tvec - mean_t
del_t = (1./np.sqrt(int_sum)) * np.sqrt(np.dot(int_trans_avg[sim.n_plt_posns-1, :], np.multiply(dev_vec, dev_vec)))
print("Pulse width of {:.2e}".format(del_t) + " s.")

# Save the spectrum:
plt.cla()
p_spectra = np.empty((sim.nch, sim.nt_frac), dtype=float)
for k in range(0, sim.nsch + 1):
    p_spectra[k+sim.nsch, :] = power_spectrum(p_trans_avg[-1, k, :])
for k in range(sim.nsch + 1, sim.nch):
    p_spectra[k - sim.nsch - 1, :] = power_spectrum(p_trans_avg[-1, k, :])
plt.imshow(np.transpose(p_spectra))
plt.savefig("SpectralRaster", dpi=160)
plt.cla()

simtime_end = time.time()

print("Total execution time " + str(simtime_end-simtime_start) + " s.")
