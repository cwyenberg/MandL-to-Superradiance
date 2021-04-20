# CLASSES FOR TD MB SIMULATIONS

# Pull in some functions:
from SRwVelTD_Fns import *
import SRwVelTD_RKFns


class AtomFieldProfile:
    # A class for a profile of the atoms + field

    # Assign methods:
    int_comp = int_comp                 # Intensity computation
    z_step = SRwVelTD_RKFns.z_step     # RK z-step method
    t_step = SRwVelTD_RKFns.t_step     # RK t-step method

    def prop_z(self, sim):
        # Perform a z-propagation; stepping each z with an RK algorithm:
        for z in range(1, sim.nz):
            self.z_step(z, sim)

    def __init__(self, sim, n_init=0.0, p_init=0.0, force_propz=False):

        # Copy in helpful values:
        self.nz = sim.nz
        self.nsch = sim.nsch
        self.nch = sim.nch

        # Allocate the inversions and polarisations:

        # Allocate Inversions:
        self.nkp_re = np.full((sim.nch, sim.nz), n_init, dtype=float)
        self.nkp_im = np.zeros((sim.nch, sim.nz), dtype=float)

        # Allocate Polarisations:
        if p_init != 0.0:

            if not sim.rand_things:
                self.pkp_re = np.full((sim.nch, sim.nz), p_init, dtype=float)
                self.pkp_im = np.zeros((sim.nch, sim.nz), dtype=float)
            else:
                self.pkp_re = np.empty((sim.nch, sim.nz), dtype=float)
                self.pkp_im = np.empty((sim.nch, sim.nz), dtype=float)
                for k in range(0, sim.nch):
                    self.pkp_re[k, :].fill(p_init*np.cos(sim.rand_phases[k]))
                    self.pkp_im[k, :].fill(p_init*np.sin(sim.rand_phases[k]))

        else:
            self.pkp_re = np.zeros((sim.nch, sim.nz), dtype=float)
            self.pkp_im = np.zeros((sim.nch, sim.nz), dtype=float)

        # Allocate the field data
        self.ep_re = np.empty(sim.nz, dtype=float)
        self.ep_im = np.empty(sim.nz, dtype=float)

        # Assign the field at the entry boundary point:
        tcur_ind = int(sim.t_cur / sim.dt)
        self.ep_re[0] = np.real(sim.E0_trans[tcur_ind])
        self.ep_im[0] = np.imag(sim.E0_trans[tcur_ind])

        # Allocate additional info not needed to simulate:
        self.int_tot = np.empty(sim.nz, dtype=float)

        # Propagate E if requested
        if force_propz:
            self.prop_z(sim)


class SimSetup:
    # Simulation Setup Object
    # Used to store all sim parameters and keep track of some changing values (ex: pumps) during sim

    # Universal constants:
    hbar = 1.05457e-30  # kg cm^2 s^-1 : Planck's reduced constant
    eps0 = 8.85419e-18  # C^2 s^2 kg^-1 cm^-3 : Permittivity of free space
    c_light = 2.99792e10  # cm s^-1 : Speed of light

    def gam_eval(self, t):
        self.gam_npr_cur = self.gam_npr0
        if self.gam_npr1 != 0.:
            for rep in range(0, self.pump_nrep):
                pump_OS = self.gam_n_tau0 + float(rep) * self.pump_Trep
                self.gam_npr_cur += self.gam_npr1 / (np.cosh((t - pump_OS) / self.gam_n_tp)) ** 2

        # The "Bloch Pump" is introduced to force the development of a Bloch angle as the
        #   inversion is pumped, due to interactions with the vacuum EM quantum fluctuations.
        # In short, we are "pumping the tipping angle" consistently with the initial tipping angle
        #   introduced for an initial inversion condition via the Gross and Haroche derivation.

        # NOTE: SIN IS CARRIED ELSEWHERE IN THE RK TERM EXPRESSIONS,
        #   SINCE ANGLES VARY ACROSS POSITIONS IF RANDOMIZED
        if self.gam_p_bloch_en:
            self.gam_p_bloch = self.dip * self.gam_npr_cur
        else:
            self.gam_p_bloch = 0.

    # Randomization function:
    def randomize(self):
        self.rand_phases = 2.0 * np.pi * np.random.rand(self.nch)
        randtheta0s = False
        if randtheta0s:
            self.rand_theta0s = (0.5 + np.random.rand(self.nch)) * 2.0 / np.sqrt(self.natoms / self.num_ind_samples)
        else:
            self.rand_theta0s = np.full(self.nch, 2.0 / np.sqrt(self.natoms / self.num_ind_samples))

    # Class constructor:
    def __init__(self,
                 nz, nt, nsch,
                 t_dur, len, wid, at_density,
                 dip, w0, t1, t2, gam_p_bloch_en,
                 n0, gam_n0, gam_n1, gam_n_tp, gam_n_tau0,
                 fvtype, v_sep,
                 rand_things, n_plt_posns, E0):

        # Map the inputs into the class elements:
        self.nz = nz
        self.nt = nt
        self.nsch = nsch
        self.t_dur = t_dur
        self.len = len
        self.wid = wid
        self.at_density = at_density
        self.dip = dip
        self.w0 = w0
        self.t1 = t1
        self.t2 = t2
        self.gam_p_bloch_en = gam_p_bloch_en
        self.n0 = n0
        self.gam_n0 = gam_n0
        self.gam_n1 = gam_n1
        self.gam_n_tp = gam_n_tp
        self.gam_n_tau0 = gam_n_tau0
        self.t_cur = 0.
        self.n_plt_posns = n_plt_posns
        self.E0 = E0
        self.rand_things = rand_things
        self.fvtype = fvtype
        self.v_sep = v_sep

        # Compute some constructs:
        self.natoms = self.len * np.pi * self.wid * self.wid * self.at_density
        self.dv = (2. * np.pi / self.t_dur) * self.c_light / self.w0
        if self.fvtype == 'plateau':
            self.nch = 2 * self.nsch + 1
        elif self.fvtype == 'twoplateau':
            self.nch = 4 * self.nsch + 2
            self.delta_v = float(self.v_sep) * self.dv
        self.v_width = float(2 * self.nsch + 1) * self.dv
        # (Note: if in two plateau mode, v_width is the width of one plateau only)

        # Pumps
        self.gam_npr0 = .5 * self.gam_n0           # npr stands for "N primed", which is 0.5 the inversion
        self.gam_npr1 = .5 * self.gam_n1

        # Omega differential:
        self.domega = self.dv * self.w0 / self.c_light
        # Time differential:
        self.dt = self.t_dur / (self.nt - 1)
        # z differential:
        self.dz = self.len / (self.nz - 1)

        # Construct the velocity distribution array and velocity values array:
        # Allocate velocities:
        self.vels = np.empty(self.nch, dtype=float)
        # Allocate distribution:
        self.fv = np.empty(self.nch, dtype=float)
        for index in range(0, self.nsch + 1):
            if fvtype == 'plateau':
                self.fv.fill(1. / self.v_width)
                # Assign distribution values and velocity values
                self.vels[index] = self.dv * float(index)
                self.vels[-index] = -self.dv * float(index)

            elif fvtype == 'twoplateau':
                # Assign distribution values and velocity values
                self.fv.fill(.5 / self.v_width)
                # Left plateau velocities:
                self.vels[self.nsch - index] = -.5 * self.delta_v - self.dv * float(index)
                self.vels[self.nsch + index] = -.5 * self.delta_v + self.dv * float(index)
                # Right plateau velocities:
                self.vels[3 * self.nsch + 1 - index] = .5 * self.delta_v - self.dv * float(index)
                self.vels[3 * self.nsch + 1 + index] = .5 * self.delta_v + self.dv * float(index)

            else:
                print("Error: Invalid Fv distribution type. NOTE: GAUSSIAN DEPRECATED IN THIS CODE VERSION.")
                raise ValueError

        # Initial Bloch angle from Gross and Haroche
        self.theta0 = 2. / np.sqrt(self.natoms)
        self.npr_init = .5 * self.n0 * np.cos(self.theta0)  # nprimed is half the inversion level
        # p_init is used as the real part of P^+, which is half the initial polarization (P=P^+ + P^-):
        self.p_init = .5 * self.dip * self.n0 * np.sin(self.theta0)

        # Randomize if requested
        if self.rand_things:
            self.randomize()    # Randomize all polarization phases and initial Bloch angle near theta_0
        else:
            # No random polarisation phases:
            self.rand_phases = np.zeros(self.nch)
            # No random tipping angle:
            self.rand_theta0s = np.full(self.nch, 2.0 / np.sqrt(self.natoms))

        # The following are time-varying values during the sim
        # (They must be computed prior to use):
        self.gam_p_cur = 0.    # No polarisation pump used in this sim, besides the so-called "Bloch pump"
        self.gam_eval(0.)      # Evaluate the value of the pumps at the starting point in time, tau=0.0

        # Initialise the E field transient with the constant incident E field at z=0
        # (This array can be generalised to a time-dependent function as desired)
        self.E0_trans = np.full(self.nt, self.E0, dtype=complex)
