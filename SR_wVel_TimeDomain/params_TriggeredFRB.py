import numpy as np

# ~~~~~~~~~~~~~~~~~~ SIMULATION CONFIGURATION PARAMETER FILE ~~~~~~~~~~~~~~~~~~~~~~

t_dur = 1.e-1                               # Simulation time duration (s).
t_sim_frac = 1.                             # Fraction of t_dur to actually execute
en_loc_popns = False                         # Enable population count by number of molecules within
                                            #  a local velocity neighbourhood defined by t_char_expected (next)
t_char_expected = 1.e7                      # Expected characteristic time of pulse (s)
                                            #  (for velocity channel interaction neighbourhood sizing)
stren_fac = 1.                              # An easy factor for adjusting a few inversion amplitude characteristics
size_fac = 1.                               # An easy factor for adjusting simulation size
pump_bw_fac = 1.                            # Adjusts the bandwidth of the pump pulse

# Simulation size parameters:
nt = max(int(size_fac * 100), 500)          # Number of time points.
nz = 501                                    # Number of z posns.
                                            #  (should be a multiple of n_plt_posns + 1)
nsch = 0                                    # Number of side channels.

# Simulation physical characteristics:
sample_len = 4.2e13                         # Sample length (m).
sample_rad = 1.58e6                         # Sample radius (m).
at_density = 1.e4                           # Atomic density (m^{-3}).
n0 = 1. * at_density                        # Initial pop inv (m^{-3}).
E0_mean_VperM = 0.                          # Incident mean electric field at z=0 (V/m).
E0_stdev_VperM = 0.                         # Incident electric field standard dev at z=0.

# Medium properties
debye = 3.33564e-30                         # C m : CONSTANT : One Debye
dip = .513 * debye                          # Dipole moment mat elt (C m).
w0 = 2. * np.pi * 1.612e9                   # Angular freq of emission (s^-1).
t1 = 10.                                    # Relaxation time constant (s).
t2 = 1.2e-3                                 # Dephasing time constant (s).

# Pump characteristics
gam_n0 = n0 / t1                            # Inversion constant pump term (m^{-3} s^{-1}).
gam_p_bloch_en = False                      # Toggle "Bloch angle pumping": to remain consistent with the SR
                                            #  initial tipping angle prescription, enable this feature so that
                                            #  the polarisation pumps along with any inversion pump, in proportion
                                            #  established by the initial tipping angle prescription;
                                            #  i.e., Lambda^(P) = Lambda^(N) * d * sin(theta_0) at all times.

p_tip_init = True                           # Initialise polarisation through tipping angle prescription

gam_inv_noise = 0.                          # Inversion pump white noise amplitude
gam_pol_noise = 0.                          # Polarisation pump white noise amplitude
                                            #  (note: these are in units of n0/t1 and p_init/t2

# Pump pulse parameters
gam_n1 = 0. * 1. * n0 / t1                  # Pump pulse amplitude (m^{-3} s^{-1}).
gam_n_tp = 5.e6 / pump_bw_fac               # Pump pulse duration (s).
gam_n_tau0 = 1.e7                           # Pump pulse delay (s).

fvtype = 'plateau'                          # Velocity distribution profile type (string).
                                            #  Options: 'plateau', 'twoplateau', 'gaussian'
tp_submode = 'both'                         # Two-plateau sub-mode. Options are:
                                            #  'both' : both left and right plateaus enabled
                                            #  'left' : only left plateau enabled
                                            #  'right' : only right plateau enabled

v_sep = 24.                                 # Applies only if fvtye=='twoplateau'. Separation of two plateaus
                                            #  in units of fundamental velocity differential dv=(2pi/T)c/omega_0

# Random settings
rand_things = False                         # Toggle randomisation of polarisation phases and initial tipping angles
n_rand_runs = 1                             # Number of multiple runs to do for averaging.
en_rand_polinit = False                     # Enable randomised timing of polarisation initiation
                                            #  as per a survival model of characteristic time t_classical

# Visualisation details:
animate = True                              # Toggle to animate polarisation as simulation executes.
plotstep = 10                               # Number of time steps between visualisation frames of unfolding sim
                                            #  and between progress annunciation.
bandstep = max(int(nsch / 8), 1)            # Visualise every bandstep^th velocity channel. i.e., if 10, only draw
                                            #   every tenth velocity channel when animating unfolding sim.
n_plt_posns = 4                             # Number of z-positions higher than z=0.0L to store transients of

smoothing_bw = 50                           # Bandwidth for smoothing filter
coh_bw = 100                                # Bandwidth to check for coherence within

# Define the E0 transient
E0_mean = 0.                                # Incident mean electric field at z=0 (V/m).
                                            #  (not used in this param file).
E0_stdev = 0.                               # Incident electric field noise standard dev at z=0
                                            #  (not used in this param file).
E0_pulse_amp = 3.2e-10j                     # Incident electric field pulse amplitude (V/m)
E0_pulse_time = 4.85e-2                     # Incident electric field pulse delay time (s)
E0_pulse_width = 5.4e-4                     # Incident electric field pulse width (s)
