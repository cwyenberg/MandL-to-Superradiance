# TIME DOMAIN RUNGE-KUTTA FUNCTIONS

# The mathematics of this file are not commented;
#   see the paper's equations (6)--(8) as well as 4th Order Runge Kutta theory

# Import things
from SRwVelTD_Fns import *
import SRwVelTD_Classes


def z_step(self, z, sim):
    # Runge-Kutta step to this z position from z-1, this is a method on the Atom+Field class

    # RK1 is computed using the slope at the start:
    pkp_atdpt = (self.pkp_re[:,z-1], self.pkp_im[:,z-1])
    ep_prev = (self.ep_re[z-1], self.ep_im[z-1])
    (ep_re_rk1, ep_im_rk1) = rk_z_term(pkp_atdpt, ep_prev, sim)

    # RK2 is computed using the slope at the midpoint using the RK1 result;
    # However, in fact, the RK1 result is not needed, as the unknown
    # E field we are propagating is not part of the forcing side.
    pkp_re_atdpt = .5 * (self.pkp_re[:,z-1] + self.pkp_re[:,z])
    pkp_im_atdpt = .5 * (self.pkp_im[:,z-1] + self.pkp_im[:,z])
    pkp_atdpt = (pkp_re_atdpt, pkp_im_atdpt)
    (ep_re_rk2, ep_im_rk2) = rk_z_term(pkp_atdpt, ep_prev, sim)

    # (RK3 is, in this problem, the same as RK2; so not needed)

    # RK4 is computed using the slope at the endpoint:
    pkp_atdpt = (self.pkp_re[:, z], self.pkp_im[:, z])
    (ep_re_rk4, ep_im_rk4) = rk_z_term(pkp_atdpt, ep_prev, sim)

    self.ep_re[z] = (ep_re_rk1 + 4.*ep_re_rk2 + ep_re_rk4) / 6.
    self.ep_im[z] = (ep_im_rk1 + 4.*ep_im_rk2 + ep_im_rk4) / 6.


def rk_z_term(pkp_atdpt, ep_prev, sim):
    # RK z term computation, this is a method on the Atom+Field class

    # "atdpt" denotes "at derivative point" - i.e. value of something at the
    # point at which this RK term's derivative is computed

    # Expand the tuples passed:
    (pkp_re_atdpt, pkp_im_atdpt) = pkp_atdpt
    (ep_re_prev, ep_im_prev) = ep_prev

    # A useful float:
    tempf_c = sim.dz * sim.w0 * sim.dv / (2. * sim.eps0 * sim.c_light)

    # Compute the RK term:
    ep_re_nxt = ep_re_prev + tempf_c * np.dot(sim.fv, pkp_im_atdpt)
    ep_im_nxt = ep_im_prev + tempf_c * np.dot(sim.fv, pkp_re_atdpt)

    # Return the values
    return ep_re_nxt, ep_im_nxt


def t_step(self, sim):
    # RK time step operation, this is a method on the Atom+Field class

    t_float = sim.t_cur
    sim.gam_eval(t_float)
    atm_fld_rk1 = rk_t_term(self, self, sim, t_float)

    t_float = sim.t_cur + .5 * sim.dt
    sim.gam_eval(t_float)
    atm_fld_rk2_in = atmfld_lin_comb(.5, self, .5, atm_fld_rk1, sim)
    atm_fld_rk2 = rk_t_term(atm_fld_rk2_in, self, sim, t_float)

    atm_fld_rk3_in = atmfld_lin_comb(.5, self, .5, atm_fld_rk2, sim)
    atm_fld_rk3 = rk_t_term(atm_fld_rk3_in, self, sim, t_float)

    t_float = sim.t_cur + sim.dt
    sim.gam_eval(t_float)
    atm_fld_rk4 = rk_t_term(atm_fld_rk3, self, sim, t_float)

    atm_fld_tmp = atmfld_lin_comb(1./6., atm_fld_rk1, 1./3., atm_fld_rk2, sim)
    atm_fld_tmp = atmfld_lin_comb(1., atm_fld_tmp, 1./3., atm_fld_rk3, sim)
    atm_fld_tmp = atmfld_lin_comb(1., atm_fld_tmp, 1./6., atm_fld_rk4, sim)

    # If the random polarisation initiation is enabled, it must be executed here:
    # Conduct random polarisation initiation through a pump term, if enabled
    if sim.en_rand_polinit:
        init_prob = 1. - np.exp(-sim.dt / sim.t_classical)
        for ch, z in np.ndindex(sim.nch, sim.nz):
            if not sim.has_polarised[ch, z]:
                if np.random.rand() < init_prob:
                    atm_fld_tmp.pkp_re[ch, z] += sim.p_init[ch, z].real
                    atm_fld_tmp.pkp_im[ch, z] += sim.p_init[ch, z].imag
                    sim.has_polarised[ch, z] = True
        atm_fld_tmp.prop_z(sim)

    # Assign:
    self.nkp_re = atm_fld_tmp.nkp_re
    self.nkp_im = atm_fld_tmp.nkp_im
    self.pkp_re = atm_fld_tmp.pkp_re
    self.pkp_im = atm_fld_tmp.pkp_im
    self.ep_re = atm_fld_tmp.ep_re
    self.ep_im = atm_fld_tmp.ep_im


def rk_t_term(atm_fld_atdpt, atm_fld_prev, sim, t_float):
    # RK t term computation, this is a function on two Atom+Field objects

    # Allocate:
    atm_fld_rk = SRwVelTD_Classes.AtomFieldProfile(sim)

    # Helpful float constants:
    tempf_a = -2. * sim.dt / sim.hbar
    tempf_b = sim.dt * sim.w0 / sim.c_light
    tempf_c = 2. * sim.dt * sim.dip * sim.dip / sim.hbar

    # RK forward in time:
    for k in range(0, sim.nch):

        # Grab the velocity array in a simpler name:
        vel = sim.vels[k]

        # Inversion has only a real part:
        inv_noise = sim.gam_inv_noise * (1. - 2. * np.random.rand())
        atm_fld_rk.nkp_re[k, :] = atm_fld_prev.nkp_re[k,:] + \
                                 tempf_a * (np.multiply(atm_fld_atdpt.ep_re, atm_fld_atdpt.pkp_im[k,:]) +
                                            np.multiply(atm_fld_atdpt.ep_im, atm_fld_atdpt.pkp_re[k,:])) - \
                                 (sim.dt/sim.t1) * atm_fld_atdpt.nkp_re[k,:] + \
                                 sim.dt * (sim.gam_npr_cur + inv_noise)

        # Polarisations; the Bloch pump term mimics spontaneous emission using the initial tipping angle of the sample
        pump_rot_angle = sim.w0 * (vel / sim.c_light) * t_float  # The Bloch pump is Doppler shifted

        # Add noise to the pump if requested
        stdevs = sim.gam_p_decoh_stdev * np.abs(atm_fld_rk.nkp_re[k, :]) / sim.n0
        gam_p_decoh = np.multiply(np.random.normal(loc=0., scale=stdevs, size=sim.nz),
                                    np.exp(2.j * np.pi * np.random.rand(sim.nz)))

        # Randomly phase the pump
        rand_phases_tmp = 2.*np.pi*np.random.rand(sim.nz)

        atm_fld_rk.pkp_re[k, :] = atm_fld_prev.pkp_re[k, :] - tempf_b * vel * atm_fld_atdpt.pkp_im[k, :] + \
                                 tempf_c * np.multiply(atm_fld_atdpt.ep_im, atm_fld_atdpt.nkp_re[k, :]) - \
                                 (sim.dt/sim.t2) * atm_fld_atdpt.pkp_re[k, :] + sim.dt * (
                                         sim.gam_p_bloch *
                                         np.multiply(np.cos(pump_rot_angle+rand_phases_tmp),
                                                     np.sin(sim.rand_theta0s[k, :]))
                                         + gam_p_decoh.real)

        atm_fld_rk.pkp_im[k, :] = atm_fld_prev.pkp_im[k, :] + tempf_b * vel * atm_fld_atdpt.pkp_re[k, :] + \
                                 tempf_c * np.multiply(atm_fld_atdpt.ep_re, atm_fld_atdpt.nkp_re[k, :]) - \
                                 (sim.dt/sim.t2) * atm_fld_atdpt.pkp_im[k, :] + sim.dt * (
                                         sim.gam_p_bloch *
                                         np.multiply(np.sin(pump_rot_angle+rand_phases_tmp),
                                                     np.sin(sim.rand_theta0s[k, :]))
                                         + gam_p_decoh.imag)

    # Propagate the electric field in z:
    atm_fld_rk.prop_z(sim)

    return atm_fld_rk
