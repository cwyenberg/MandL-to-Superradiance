#################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~ SR_wVel_FourierDomain_mp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ FOURIER DOMAIN SIMULATION OF THE VELOCITY DEPENDENT MB EQUATIONS ~~~~~~~
# (MP DENOTES WITH MULTIPROCESSING)
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
# This code can operate in multiple modes:
#   CONVENTIONAL MENEGOZZI AND LAMB ALGORITHM
#   TRANSIENT MENEGOZZI AND LAMB ALGORITHM (refer to the paper for terminology)
#   INTEGRAL FOURIER ALGORITHM (WITH OR WITHOUT MULTIPROCESSING)
#   (Plus others not documented in the paper)
#
# SWITCHING BETWEEN MODES TAKES SOME DOING:
#   (1) TO SELECT BETWEEN IF AND CONVENTIONAL M&L:
#       - NAVIGATE TO 'SRwVelFD_Classes.py',
#       - ADJUST COMMENTING TO MAP THE PROPER SOLUTION METHOD TO THE CLASS METHOD DEFINITION
#       - NEAR THE TOP OF THE CLASS DEFINITION (SEE COMMENTS THERE FOR CLARITY)
#   (2) IF WORKING WITH CONVENTIONAL M&L, CHOOSE BETWEEN TRANSIENT AND STEADY-STATE ALGORITHMS BY:
#       - NAVIGATE TO 'SRwVelFD_ConvMLFns.py'
#       - ASSIGN THE MODE CONSTANT TO YOUR CHOICE BETWEEN 1 AND 4 (SEE COMMENTS THERE)
#
# CHANGE THE PICKLE FILENAMES ON LINES (211)--(216) ACCORDING TO THE MODE OF OPERATION
#
# Simulation parameters required to recover paper figures provided in comments
# and flagged as PAPER_REF: NNN
#
#################################################################################

from SRwVelFD_Classes import *
import pickle

# Multiprocessing conditioner:
if __name__ == '__main__':

    # Import libraries
    import time
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    import numpy as np

    # ~~~~~~~~~~~~~~~~~~ SIMULATION CONFIG DATA ~~~~~~~~~~~~~~~~~~~~~~

    t_dur = 1.e8                            # Simulation time duration (s).  PAPER_REF: 1.e8

    # Simulation Numerical Size
    nt = 500                                # Number of time points.  PAPER_REF: 500
    nz = 41                                 # Number of z posns.  PAPER_REF: 41
                                            #   should be (multiple of n_plt_posns) + 1
    nsch = 10                               # Number of side channels.  PAPER_REF: 10

    # Sim fourier fidelity
    n_side_modes = 50                       # Number of side modes in each vel channel fs.  PAPER_REF: 50
    nint = 30                               # LMI Truncation Nint.  PAPER_REF: 30
                                            #   NOTE: IF CHANGING THIS VALUE,
                                            #   YOU MUST ALSO CHANGE PICKLE FILENAMES BELOW

    # Atomic properties
    at_density = 1.5e-12                    # Atomic number density (cm^{-3}).  PAPER_REF: 1.5e-12
    n0 = 1. * at_density                    # Initial inversion density (cm^{-3}).  PAPER_REF: 1. * at_density

    # Medium properties
    debye = 3.33564e-28                     # C cm : One Debye
    dip = 0.7 * debye                       # C cm : Methanol dipole moment mat elt
    w0 = 2.0*np.pi*6.7e9                    # s^-1 : Methanol transition ang freq
    t1 = 1.64e7                             # Relaxation time constant T1 (s).  PAPER_REF: 1.64e7
    t2 = 1.55e6                             # Dephasing time constant T2 (s).  PAPER_REF: 1.55e6

    E0_VperM = 1.e-16                       # Incident E field at z=0 (V/m).  PAPER_REF: 1.e-16

    # Inversion pump properties
    lam_n0 = n0 / t1                        # Inversion pump (cm^{-3}/s).  PAPER_REF: n0 / t1

    lam_n1 = 0.                             # Inversion pump pulse amplitude (cm^-3 s^-1).  PAPER_REF: 0.
    lam_n_tp = 5.e6                         # Inversion pump pulse duration (s).  PAPER_REF: 5.e6 BUT IRRELEVANT
    lam_n_tau0 = 1.e7                       # Inversion pump pulse delay (s).  PAPER_REF: 1.e7 BUT IRRELEVANT

    gam_p_bloch_en = True                   # Toggle bloch angle based pol pumping.  PAPER_REF: True

    numposns = 20                           # Number of positions for which to store transients.  PAPER_REF: 20


    # Simulation Physical Dimensions
    zlen = 2.e15                            # Sample length (cm).  PAPER_REF: 2.e15
    wid = 5.4e7                             # Sample radius (cm).  PAPER_REF: 5.4e7

    rand_things = False                     # Toggle pol phase and init tip randomization.  PAPER_REF: False

    plot_select = 3                         # Type of plot to animate during sim.
    animate = True                          # Toggle animation

    # ~~~~~~~~~~~~~~~~~~~~ END INPUT DATA ~~~~~~~~~~~~~~~~~~~

    # Select multiprocessing and annunciate:
    mp_enabled = (nsch > 1)
    if mp_enabled:
        mp.set_start_method("fork")
    print("Multiprocessing: " + str(mp_enabled))

    # Seed  the random number generator:
    np.random.seed()

    # Compute some useful values:
    E0 = 100. * E0_VperM                    # E0 with length component of units in cm (suited to this sim)

    # Simulation start time:
    simtime_start = time.time()

    # Construct the physical parameters; the class constructor will auto-calculate the pump:
    sim = SimSetup(mp_enabled, nt, nz,
                   nsch, n_side_modes, nint,
                   lam_n0, lam_n1, lam_n_tp, lam_n_tau0,
                   t_dur, zlen, wid,
                   at_density, n0, E0,
                   dip, w0, t1, t2, rand_things, gam_p_bloch_en)

    print("Simulating over a velocity width of " + str(sim.v_width) + " cm/s.")

    t_start = time.time()

    # Allocate the Atom and Field Time slice at z = 0
    z_slice = AFZSlice(sim)
    # The e field fs at z = 0 vanishes:
    z_slice.e_fs = sim.E0_fs
    # Propagate the inversion and polarization through time at z = 0:
    z_slice.compute_inv_pol_fs_from_e_fs()
    z_slice.compute_tseries()

    if plot_select == 1:
        plt.cla()
        for p in range(0, sim.n_tot_chs):
            plt.plot(np.real(z_slice.inv_tseries[p, :]), linewidth=1.)
    elif plot_select == 2:
        plt.cla()
        for p in range(0, sim.n_tot_chs):
            plt.plot(np.real(z_slice.pol_tseries[p, :]), linewidth=1.)
            plt.plot(np.imag(z_slice.pol_tseries[p, :]), linewidth=1.)
    elif plot_select == 3:
        plt.cla()
        plt.plot(np.real(np.multiply(np.conjugate(z_slice.e_tseries), z_slice.e_tseries)), linewidth=1.)

    if plot_select != 0:
        plt.show(block=False)
        plt.pause(0.01)

    # With everything initialized, proceed with the RK time stepping.

    # Allocate an array for intensity transients
    int_transients = np.empty((numposns, sim.nt), dtype=float)
    inv_transients = np.empty((numposns, sim.nt), dtype=float)
    pol_transients = np.empty((numposns, sim.nt), dtype=complex)

    # Execute the simulation
    for z in range(1, sim.nz):

        print("Computing for z = " + str(z) + "...")
        t_zstart = time.time()
        z_slice = z_slice.z_advance()
        t_zend = time.time()
        print("Took " + str(t_zend-t_zstart) + " s.")

        if animate:
            # Plot while running
            z_slice.compute_tseries()

            if plot_select == 1:
                plt.cla()
                for p in range(0, sim.n_tot_chs):
                    plt.plot(np.real(z_slice.inv_tseries[p, :]), linewidth=1.)
            elif plot_select == 2:
                plt.cla()
                for p in range(0, sim.n_tot_chs):
                    plt.plot(np.real(z_slice.pol_tseries[p, :]), linewidth=1.)
                    plt.plot(np.imag(z_slice.pol_tseries[p, :]), linewidth=1.)
            elif plot_select == 3:
                plt.cla()
                plt.plot(np.real(np.multiply(np.conjugate(z_slice.e_tseries), z_slice.e_tseries)), linewidth=1.)
            plt.show(block=False)
            plt.pause(0.01)

        z_plot_step = int((sim.nz-1)/numposns)
        if (z % z_plot_step) == 0:
            # Store the transient
            z_ind = int(z/z_plot_step) - 1
            if not animate:
                # (if animated, this was already calculated above)
                z_slice.compute_tseries()
            int_transients[z_ind, :] = np.real(np.multiply(np.conjugate(z_slice.e_tseries), z_slice.e_tseries))
            inv_transients[z_ind, :] = z_slice.inv_tseries[0, :]
            pol_transients[z_ind, :] = z_slice.pol_tseries[0, :]

    # Compute the time series representations of all quantities:
    z_slice.compute_tseries()

    t_end = time.time()

    print("Sim took " + str(t_end - t_start) + " seconds.")

    plt.cla()

    # Pickle the intensities data for later plotting
    # NOTE: CHANGE THE BELOW FILENAMES TO REFLECT THE CURRENT MODE
    with open('if_nint30_intensities.pickle', 'wb') as f:
        pickle.dump(int_transients, f)
    with open('if_nint30_inversions.pickle', 'wb') as f:
        pickle.dump(inv_transients, f)
    with open('if_nint30_polarizations.pickle', 'wb') as f:
        pickle.dump(pol_transients, f)

    # Peaks and timing:
    end_intensity = int_transients[numposns-1, :]
    peak_intensity = np.amax(end_intensity)
    peak_time = sim.t_dur * float(np.argmax(end_intensity)) / (float(len(end_intensity))-1.)

    # The width. We use a mean square deviation metric:
    tvec = np.linspace(0, sim.t_dur, num=sim.nt, endpoint=True)
    int_sum = np.sum(end_intensity)
    mean_t = np.dot(end_intensity, tvec) / int_sum
    dev_vec = tvec - mean_t
    del_t = (1./np.sqrt(int_sum)) * np.sqrt(np.dot(end_intensity, np.multiply(dev_vec, dev_vec)))

    # Annunciate some results:
    print("For a spread of " + str(sim.v_width) + " cm/s,")
    print("the peak intensity was {:.2e}".format(peak_intensity) + ";")
    print("it was reached at t = {:.2e}".format(peak_time) + " s; and")
    print("the pulse width was {:.2e}".format(del_t) + " s.")
