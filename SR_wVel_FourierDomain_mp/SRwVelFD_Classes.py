# The usual libraries
import numpy as np
from SRwVelFD_IFFns import z_advance
from SRwVelFD_IFFns import compute_inv_pol_fs_from_e_fs_if              # THESE IMPORTS USED WHEN IN APPROPRIATE MODE
from SRwVelFD_IFFns_MP import compute_inv_pol_fs_from_e_fs_if_mp        # THESE IMPORTS USED WHEN IN APPROPRIATE MODE
from SRwVelFD_ConvMLFns import compute_inv_pol_fs_from_e_fs_conv_ml     # THESE IMPORTS USED WHEN IN APPROPRIATE MODE
from SRwVelFD_IFFns import compute_tseries
from SRwVelFD_IFFns import trunc_ftx


class AFZSlice:
    # A single z-value slice of the atom + field

    # Assign class methods defined elsewhere
    z_advance = z_advance
    compute_tseries = compute_tseries

    # UNCOMMENT ONE (AND *ONLY* ONE) OF THE FOLLOWING THREE LINES TO SELECT FROM:
    #  (A) ..._if_mp = INTEGRAL FOURIER METHOD WITH MULTIPROCESSING
    #  (B) ..._conv_ml = CONVENTIONAL MENEGOZZI AND LAMB METHOD
    #  (C) OBSOLETE ...if    = INTEGRAL FOURIER METHOD WITHOUT MULTIPROCESSING
    # NOTE: IF WITHOUT MP IS OBSOLETE AND NEED NOT BE USED; CAN LEAVE ...if_mp METHOD (A) APPLIED
    #       AND TOGGLE MULTIPROCESSING VIA 'MP_ENABLED' SWITCH IN MAIN ROUTINE
    compute_inv_pol_fs_from_e_fs = compute_inv_pol_fs_from_e_fs_if_mp     # (A) UNCOMMENT FOR IF WITH MP
    # compute_inv_pol_fs_from_e_fs = compute_inv_pol_fs_from_e_fs_conv_ml   # (B) UNCOMMENT FOR CONVENTIONAL M&L
    # compute_inv_pol_fs_from_e_fs = compute_inv_pol_fs_from_e_fs_if        # (C) UNCOMMENT FOR IF WITHOUT MP

    def __init__(self, sim):
        # Class instance constructor

        # Associate the sim data
        # (need this info in the class for the __add__ and __mul__ methods)
        self.sim = sim

        # Allocate the data (Fourier series and time series)
        self.inv_fs = np.empty([sim.n_tot_chs, sim.n_side_modes + 1], dtype=complex)
        self.pol_fs = np.empty([sim.n_tot_chs, sim.n_tot_modes], dtype=complex)
        self.e_fs = np.empty(sim.n_e_tot_modes, dtype=complex)
        self.inv_tseries = np.empty([sim.n_tot_chs, sim.nt], dtype=float)
        self.pol_tseries = np.empty([sim.n_tot_chs, sim.nt], dtype=complex)
        self.e_tseries = np.empty(sim.nt, dtype=complex)

    def __add__(self, other):
        # Class addition

        # Add two class instances
        afzs_return = AFZSlice(self.sim)
        afzs_return.inv_fs = self.inv_fs + other.inv_fs
        afzs_return.pol_fs = self.pol_fs + other.pol_fs
        afzs_return.e_fs = self.e_fs + other.e_fs

        return afzs_return

    def __mul__(self, other):
        # Class multiplication by scalar

        # Multiply by a float scalar; assumes class occurs on left of * operator
        afzs_return = AFZSlice(self.sim)
        afzs_return.inv_fs = other * self.inv_fs
        afzs_return.pol_fs = other * self.pol_fs
        afzs_return.e_fs = other * self.e_fs

        return afzs_return

    def __rmul__(self, other):
        # (Right multiplication as needed if class occurs on right of * operator)
        return self * other


# Simulation Class
class SimSetup:

    # Universal constants:
    hbar = 1.05457e-30  # kg cm^2 s^-1 : Planck's reduced constant
    eps0 = 8.85419e-18  # C^2 s^2 kg^-1 cm^-3 : Permittivity of free space
    c_light = 2.99792e10  # cm s^-1 : Speed of light

    # Constructor:
    def __init__(self,
                 mp_enabled, nt, nz,
                 nsch, n_side_modes, nint,
                 lam_n0, lam_n1, lam_n_tp, lam_n_tau0,
                 t_dur, zlen, wid,
                 at_density, n0, E0,
                 dip, w0, t1, t2, rand_things, gam_p_bloch_en):

        # Map the inputs into the class elements:
        self.mp_enabled = mp_enabled
        self.pool_opened = False
        self.nt = nt
        self.nz = nz
        self.nsch = nsch
        self.n_side_modes = n_side_modes
        self.nint = nint

        self.lam_n0 = lam_n0
        self.lam_n1 = lam_n1
        self.lam_n_tp = lam_n_tp
        self.lam_n_tau0 = lam_n_tau0
        self.E0 = E0

        self.t_dur = t_dur
        self.zlen = zlen
        self.wid = wid

        self.at_density = at_density
        self.n0 = n0

        self.dip = dip
        self.w0 = w0
        self.t1 = t1
        self.t2 = t2

        self.gam_p_bloch_en = gam_p_bloch_en

        # Compute some constructs:

        # WARNING:
        if nint > n_side_modes:
            print("WARNING: TRUNCATION LENGTH EXCEEDS SIDE MODE COUNT (NSM); CROPPING TO NSM")
            self.nint = self.n_side_modes

        # Total number of atoms
        self.n_atoms = self.zlen * np.pi * self.wid * self.wid * self.at_density

        # Total number of velocity channels
        self.n_tot_chs = 2 * self.nsch + 1
        # Number of modes (of inv and pol) to simulate in each velocity channel
        self.n_tot_modes = 2 * self.n_side_modes + 1

        # Number of field modes extend past the velocity bandwidth by the mode count
        self.n_e_side_modes = self.nsch + self.n_side_modes
        self.n_e_tot_modes = 2 * self.n_e_side_modes + 1

        # Useful bounds for t and xi array work:
        self.n_xi_side_elts = self.nsch + self.nint
        self.n_xi_tot_elts = 2 * self.n_xi_side_elts + 1
        self.n_t_side_elts = self.n_xi_side_elts + self.n_e_side_modes
        self.n_t_tot_elts = 2 * self.n_t_side_elts + 1

        # Velocity profile properties
        self.domega = 2.0 * np.pi / self.t_dur                  # Fundamental frequency differential
        self.dv = self.domega * self.c_light / self.w0          # Velocity differential of adjacent channels
        self.v_width = float(self.n_tot_chs) * self.dv
        if rand_things:
            self.v_phase_mults = np.exp(1.j * 2. * np.pi * np.random.rand(self.n_tot_chs))
        else:
            self.v_phase_mults = np.full(self.n_tot_chs, 1. + 0.j)

        # NOTE: "INV" BELOW REFERS TO THE INVERSION IN THE MB EQUATIONS, WHICH IS HALF THE ACTUAL INVERSION
        #   IN THE TIME DOMAIN CODE, WE DISTINGUISH THIS WITH NOTATION '..pr' FOR "PRIMED", BUT NOT HERE
        # Pump values (primed)
        self.lam_inv0 = .5 * self.lam_n0
        self.lam_inv1 = .5 * self.lam_n1

        # Grid step sizes
        self.dz = self.zlen / (float(self.nz) - 1)               # z step

        # Construct the velocity profile:
        self.fv = np.full(self.n_tot_chs, 1. / self.v_width, dtype=float)

        # Initial Bloch angle from Gross and Haroche
        self.theta0 = 2. / np.sqrt(self.n_atoms)

        # Initial inversion (recall that we are working with nprimed = .5 n0)
        self.inv_init_re = 0.5 * self.n0 * np.cos(self.theta0)  # inv in MB eqns is half the inversion level

        # p_init is used as the real part of P^+, which is half the initial polarization (P=P^+ + P^-):
        self.pol_init = 0.5 * self.dip * self.n0 * np.sin(self.theta0)

        # For storing pump time series
        self.lam_inv_tspace = np.empty(self.nt, dtype=float)
        self.lam_pol_tspace = np.empty(self.nt, dtype=complex)

        # Compute the pumps in time
        for (index, t) in enumerate(np.linspace(0.0, t_dur, num=self.nt, endpoint=True)):
            (self.lam_inv_tspace[index], self.lam_pol_tspace[index]) = self.lam_eval(t)
        # Assign the initial z=0 E-Field in time
        self.E_tspace = np.full(self.nt, self.E0, dtype=complex)

        # Compute the pumps and E0 in Fourier domain
        self.lam_inv_fs = trunc_ftx(self.lam_inv_tspace, self.n_side_modes)
        self.lam_pol_fs = trunc_ftx(self.lam_pol_tspace, self.n_side_modes)
        self.E0_fs = trunc_ftx(self.E_tspace, self.n_e_side_modes)

    def lam_eval(self, t):
        # Calculate the pump terms at a requested moment in time
        # ("lam" stands for "Lambda", the symbol used for pumps in documentation)

        # First the n-primed pump
        lam_inv_cur = self.lam_inv0 + self.lam_inv1 / (np.cosh((t - self.lam_n_tau0) / self.lam_n_tp)) ** 2

        # The "Bloch Pump" is introduced to force the development of a Bloch angle as the
        #   inversion is pumped, due to interactions with the vacuum EM quantum fluctuations.
        # In short, we are "pumping the tipping angle" consistently with the initial tipping angle
        #   introduced for an initial inversion condition via the Gross and Haroche derivation.
        if self.gam_p_bloch_en:
            lam_p_bloch = self.dip * lam_inv_cur * np.sin(self.theta0) / np.cos(self.theta0)
        else:
            lam_p_bloch = 0.

        return lam_inv_cur, lam_p_bloch
