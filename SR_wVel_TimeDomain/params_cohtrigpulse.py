import numpy as np

# ~~~~~~~~~~~~~~~~~~ SIMULATION CONFIGURATION PARAMETER FILE ~~~~~~~~~~~~~~~~~~~~~~

anim_type = 'inv'                           # Choose what to animate; select from:
                                            #  'inv' : inversion along length;
                                            #  'pol' : real part of polarisation along length;
                                            #  'polangle' : angle of polarisation across channels
                                            #  'int' : endfire intensity cumulative history;

t_dur = 5.e7                                # Simulation time duration (s).
t_sim_frac = .5                             # Fraction of t_dur to actually execute     N/A TO THIS PAPER
en_loc_popns = False                        # Enable population count by number of molecules within
                                            #  a local velocity neighbourhood defined by t_char_expected (next)
t_char_expected = 1.e7                      # Expected characteristic time of pulse (s) N/A TO THIS PAPER
                                            #  (for velocity channel interaction neighbourhood sizing)
stren_fac = 10.                            # An easy factor for adjusting a few inversion amplitude characteristics
                                            #  N/A TO THIS PAPER
size_fac = 10                                # An easy factor for adjusting simulation size
                                            #  N/A TO THIS PAPER
pump_bw_fac = 1.                            # Adjusts the bandwidth of the pump pulse N/A TO THIS PAPER

# Simulation size parameters:
nt = 1000                                   # Number of time points.
nz = 401                                    # Number of z posns.
                                            #  (should be a multiple of n_plt_posns + 1)
nsch = int(size_fac * 10)                                   # Number of side channels.

# Simulation physical characteristics:
sample_len = 2.e13                          # Sample length (m).
sample_rad = 5.4e5                          # Sample radius (m).
at_density = stren_fac * 3.e-6                         # Atomic density (m^{-3}).
n0 = 1. * at_density                        # Initial pop inv (m^{-3}).

# Medium properties
debye = 3.33564e-30                         # C m : CONSTANT : One Debye
dip = .7 * debye                            # Dipole moment mat elt (C m).
w0 = 2. * np.pi * 6.7e9                     # Angular freq of emission (s^-1).
t1 = 1.64e7                                 # Relaxation time constant (s).
t2 = 1.55e6                                 # Dephasing time constant (s).

# Incident electric field at z=0
E0_mean = 0.                                # Incident electric field mean at z=0 (V/m).
E0_stdev = 0.                               # Incident electric field noise standard dev at z=0
                                            #  N/A TO THIS PAPER
E0_pulse_amp = 0. * 2.e-16                           # Incident electric field pulse amplitude (V/m) N/A TO THIS PAPER.
E0_pulse_time = 1.e7                     # Incident electric field pulse delay time (s) N/A TO THIS PAPER
E0_pulse_width = 2.e5                     # Incident electric field pulse width (s) N/A TO THIS PAPER

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
gam_n1 = 0. * 1. * n0 / t1                  # Pump pulse amplitude (m^{-3} s^{-1}). N/A TO THIS PAPER
gam_n_tp = 5.e6 / pump_bw_fac               # Pump pulse duration (s). N/A TO THIS PAPER
gam_n_tau0 = 1.e7                           # Pump pulse delay (s). N/A TO THIS PAPER

fvtype = 'plateau'                       # Velocity distribution profile type (string).
                                            #  Options: 'plateau', 'twoplateau', 'gaussian'
tp_submode = 'both'                         # Two-plateau sub-mode. Options are:
                                            #  'both' : both left and right plateaus enabled
                                            #  'left' : only left plateau enabled
                                            #  'right' : only right plateau enabled

v_sep = 100                                 # Applies only if fvtye=='twoplateau'. Separation of two plateaus
                                            #  in units of fundamental velocity differential dv=(2pi/T)c/omega_0

# Random settings
rand_things = True                          # Toggle randomisation of polarisation phases and initial tipping angles
n_rand_runs = 1                             # Number of multiple runs to do for averaging.
en_rand_polinit = False                     # Enable randomised timing of polarisation initiation
                                            #  as per a survival model of characteristic time t_classical

# Visualisation details:
animate = True                              # Toggle to animate polarisation as simulation executes.
plotstep = 10                               # Number of time steps between visualisation frames of unfolding sim
                                            #  and between progress annunciation.
bandstep = max(int(nsch / 40), 1)          # Visualise every bandstep^th velocity channel. i.e., if 10, only draw
                                            #  every tenth velocity channel when animating unfolding sim.
                                            # (Only applies if the user uncomments the code within the simulation
                                            #  loop which enables per-channel inversion or polarisation animation)
n_plt_posns = 4                             # Number of z-positions higher than z=0.0L to store transients of

smoothing_bw = 50                           # Bandwidth for smoothing filter
coh_bw = 100                                # Bandwidth to check for coherence within
