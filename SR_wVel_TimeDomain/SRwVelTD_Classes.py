# CLASSES FOR TD MB SIMULATIONS

# Pull in some functions:
from SRwVelTD_Fns import *
import SRwVelTD_RKFns
import warnings

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

    def __init__(self, sim, init=False, force_propz=False):

        # Copy in helpful values:
        self.nz = sim.nz
        self.nsch = sim.nsch
        self.nch = sim.nch

        # Allocate the inversions and polarisations:

        # Allocate Inversions:
        if init:
            self.nkp_re = np.full((sim.nch, sim.nz), sim.npr_init, dtype=float)
        else:
            self.nkp_re = np.zeros((sim.nch, sim.nz), dtype=float)
        self.nkp_im = np.zeros((sim.nch, sim.nz), dtype=float)

        # Allocate Polarisations:
        if init and not sim.en_rand_polinit:
            self.pkp_re = np.real(sim.p_init)
            self.pkp_im = np.imag(sim.p_init)

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
    hbar = 1.05457e-34      # kg m^2 / s : Planck's reduced constant
    eps0 = 8.85419e-12      # C^2 s^2 / (kg m^3) : Permittivity of free space
    c_light = 2.99792e8     # m/s : Speed of light

    def gam_eval(self, t):
        self.gam_npr_cur = self.gam_npr0
        if self.gam_npr1 != 0.:
            self.gam_npr_cur += self.gam_npr1 / (np.cosh((t - self.gam_n_tau0) / self.gam_n_tp)) ** 2

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
        self.rand_phases = 2. * np.pi * np.random.rand(self.nch, self.nz)
        randtheta0s = True
        if randtheta0s:
            # Distribution is P(alpha^2) = (1/N) e^{-alpha^2/N}.
            # theta_i = (2/N) alpha.
            # Refer to Gross and Haroche
            # But here we use N_perzslicinteracting
            nat_per_zsl_inting = self.natoms_per_zslice / self.n_interacting_vwidths
            rand_alphas = np.sqrt(-nat_per_zsl_inting * np.log(1.-np.random.rand(self.nch, self.nz)))
            self.rand_theta0s = (2./nat_per_zsl_inting) * rand_alphas
        else:
            self.rand_theta0s = np.full((self.nch, self.nz), 2. / np.sqrt(self.natoms))

    # Class constructor:
    def __init__(self, params):

        # Map the inputs into the class elements:
        self.nz = params.nz
        self.nt = params.nt
        self.nsch = params.nsch
        self.t_dur = params.t_dur
        self.t_sim_frac = params.t_sim_frac
        self.t_dur_frac = self.t_sim_frac * self.t_dur
        self.nt_frac = int(self.nt * self.t_sim_frac)
        self.sample_len = params.sample_len
        self.sample_rad = params.sample_rad
        self.at_density = params.at_density
        self.dip = params.dip
        self.w0 = params.w0
        self.t1 = params.t1
        self.t2 = params.t2
        self.gam_p_bloch_en = params.gam_p_bloch_en
        self.p_tip_init = params.p_tip_init
        self.n0 = params.n0
        self.gam_n0 = params.gam_n0
        self.gam_n1 = params.gam_n1
        self.gam_n_tp = params.gam_n_tp
        self.gam_n_tau0 = params.gam_n_tau0
        self.gam_inv_noise = params.gam_inv_noise
        self.gam_pol_noise = params.gam_pol_noise
        self.t_cur = 0.
        self.n_plt_posns = params.n_plt_posns
        self.E0_mean = params.E0_mean
        self.E0_stdev = params.E0_stdev
        self.E0_pulse_amp = params.E0_pulse_amp
        self.E0_pulse_time = params.E0_pulse_time
        self.E0_pulse_width = params.E0_pulse_width
        self.rand_things = params.rand_things
        self.en_rand_polinit = params.en_rand_polinit
        self.fvtype = params.fvtype
        self.tp_submode = params.tp_submode
        self.v_sep = params.v_sep
        self.en_loc_popns = params.en_loc_popns
        self.t_char_expected = params.t_char_expected      # Expected characteristic timescale of an isolated channel

        # Expected local interaction zone
        self.delta_v_eff = 2. * np.pi * self.c_light / (self.t_char_expected * self.w0)

        # Compute some useful values:
        self.k0 = self.w0 / self.c_light        # spont em wavenumber units of cm^-1
        self.gam_spont = (self.k0**3) * (self.dip**2) / (3. * np.pi * self.eps0 * self.hbar)   # Spont em rate
        self.wavelen = 2. * np.pi / self.k0     # spont em wavelength
        self.mu = (3. * self.wavelen**2) / (8. * (np.pi**2) * (self.sample_rad**2))    # mu from Gross and Haroche
        self.natoms = self.sample_len * np.pi * (self.sample_rad**2) * self.at_density  # total num atoms
        self.dv = (2. * np.pi / self.t_dur) * self.c_light / self.w0    # vel differential
        if (self.fvtype == 'twoplateau') and (self.tp_submode == 'both'):
            self.nch = 4 * self.nsch + 2
        else:
            self.nch = 2 * self.nsch + 1
        if self.fvtype == 'twoplateau':
            self.delta_v = float(self.v_sep) * self.dv
        self.v_width = float(2 * self.nsch + 1) * self.dv
        # (Note: if in two plateau mode, v_width is the width of one plateau only)
        self.n_interacting_vwidths = self.v_width / self.delta_v_eff
        self.natoms_per_zslice = float(self.natoms) / float(self.nz)

        # Effective number of interacting atoms
        if self.en_loc_popns:
            self.natoms_eff = self.natoms * self.delta_v_eff / self.v_width
        else:
            self.natoms_eff = self.natoms

        # Time of onset of classical trajectory
        self.t_classical = 1. / (self.natoms_eff * self.gam_spont * self.mu)

        # Omega differential:
        self.domega = self.dv * self.w0 / self.c_light

        # Time differential:
        self.dt = self.t_dur / (self.nt - 1)

        # z differential:
        self.dz = self.sample_len / (self.nz - 1)

        # Pumps
        self.gam_npr0 = .5 * self.gam_n0           # npr stands for "N primed", which is .5 the inversion
        self.gam_npr1 = .5 * self.gam_n1

        # Decoherent polarisation pump standard deviation
        self.gam_p_decoh_stdev = 0. * self.dip * self.n0 \
                                 * np.sqrt(self.dt / self.t_classical) \
                                 * np.sqrt(self.dz / self.sample_len) \
                                 / (np.sqrt(self.natoms_eff) * self.t_classical)

        # Construct the velocity distribution array and velocity values array:
        # Allocate velocities:
        self.vels = np.empty(self.nch, dtype=float)
        # Allocate distribution:
        self.fv = np.empty(self.nch, dtype=float)
        for index in range(0, self.nsch + 1):
            if self.fvtype == 'plateau':
                self.fv[index] = 1. / self.v_width
                self.fv[-index] = 1. / self.v_width
                self.vels[index] = self.dv * float(index)
                self.vels[-index] = -self.dv * float(index)
                # ENABLE BELOW CODE FOR A COMB DISTRIBUTION:
                # skip = 20
                # if index % skip >= int(skip/2):
                #     self.fv[index] = 0.
                #     self.fv[-index] = 0.

            elif (self.fvtype == 'twoplateau') and (self.tp_submode == 'left'):
                self.fv[index] = .5 / self.v_width
                self.fv[-index] = .5 / self.v_width
                self.vels[index] = -.5 * self.delta_v + self.dv * float(index)
                self.vels[-index] = -.5 * self.delta_v - self.dv * float(index)

            elif (self.fvtype == 'twoplateau') and (self.tp_submode == 'right'):
                self.fv[index] = .5 / self.v_width
                self.fv[-index] = .5 / self.v_width
                self.vels[index] = .5 * self.delta_v + self.dv * float(index)
                self.vels[-index] = .5 * self.delta_v - self.dv * float(index)

            elif (self.fvtype == 'twoplateau') and (self.tp_submode == 'both'):
                # Assign distribution values and velocity values
                self.fv[self.nsch - index] = .5 / self.v_width
                self.fv[self.nsch + index] = .5 / self.v_width
                self.fv[3 * self.nsch + 1 - index] = .5 / self.v_width
                self.fv[3 * self.nsch + 1 + index] = .5 / self.v_width
                # Left plateau velocities:
                self.vels[self.nsch - index] = -.5 * self.delta_v - self.dv * float(index)
                self.vels[self.nsch + index] = -.5 * self.delta_v + self.dv * float(index)
                # Right plateau velocities:
                self.vels[3 * self.nsch + 1 - index] = .5 * self.delta_v - self.dv * float(index)
                self.vels[3 * self.nsch + 1 + index] = .5 * self.delta_v + self.dv * float(index)

            elif (self.fvtype == 'gaussian'):
                self.fv[index] = np.exp(-(index/(self.nsch/2))**2) / (self.dv * self.nsch/2 * np.sqrt(np.pi))
                self.fv[-index] = self.fv[index]
                self.vels[index] = self.dv * float(index)
                self.vels[-index] = -self.dv * float(index)

            else:
                print("Error: Invalid Fv distribution type. NOTE: GAUSSIAN DEPRECATED IN THIS CODE VERSION.")
                raise ValueError

        if self.en_rand_polinit:
            # For tracking random initial polarisation event
            self.has_polarised = np.full((self.nch, self.nz), False)

        # Initial Bloch angle from Gross and Haroche
        self.theta0 = 2. / np.sqrt(self.natoms_eff)
        self.npr_init = .5 * self.n0 * np.cos(self.theta0)  # nprimed is half the inversion level

        # Randomize if requested
        if self.rand_things:
            self.randomize()    # Randomize all polarisation phases and initial Bloch angle near theta_0
        else:
            # No random polarisation phases:
            self.rand_phases = np.zeros((self.nch, self.nz))
            # No random tipping angle:
            self.rand_theta0s = np.full((self.nch, self.nz), 2. / np.sqrt(self.natoms_eff))

        # p_init is used as P^+, which is half the initial polarization (P=P^+ + P^-):
        if self.p_tip_init:
            self.p_init = .5 * self.dip * self.n0 * np.sin(self.rand_theta0s) * np.exp(1.j * self.rand_phases)
        else:
            self.p_init = np.zeros((self.nch, self.nz), dtype=complex)

        # The following are time-varying values during the sim
        # (They must be computed prior to use):
        self.gam_p_cur = 0.    # No polarisation pump used in this sim, besides the so-called "Bloch pump"
        self.gam_eval(0.)      # Evaluate the value of the pumps at the starting point in time, tau=0.0

        # Construct the E field transient from the requested incident E field (at z=0) parameters (V/m)
        times = np.linspace(0., self.t_dur, self.nt)
        warnings.filterwarnings("ignore")
        self.E0_trans = self.E0_pulse_amp / (
                np.cosh((times - self.E0_pulse_time) / self.E0_pulse_width) ** 2.) \
                + np.random.normal(loc=self.E0_mean, scale=self.E0_stdev, size=self.nt)
