import numpy as np
import SRwVelFD_Classes


def compute_inv_pol_fs_from_e_fs_if(self):
    # Solves for inversion and polarisation Fourier series
    # from given electric field Fourier series
    # USES INTEGRAL FOURIER METHOD WITHOUT MULTIPROCESSING
    # (SHOULD NOT NEED TO USE THIS FUNCTION AT ALL)

    # Pull out the simulation  data name for convenience
    sim = self.sim

    # A useful quantities:
    nonzero_range = np.concatenate((np.arange(-sim.n_side_modes, 0), np.arange(1, sim.n_side_modes+1)))
    nonzero_range_trunc = np.concatenate((np.arange(-sim.nint, 0), np.arange(1, sim.nint + 1)))
    dipfloat = 2. * sim.dip * sim.dip

    # First compute the tm vector

    # TBD: CONSIDER USING AN FFT TO PERFORM THIS CONVOLUTION;
    #   ALGORITHM IS NOT BIG_O(N) BECAUSE OF THESE LINES:
    tm = np.empty(sim.n_t_tot_elts, dtype=complex)
    tm[0] = np.pi
    for mprime in range(1, sim.n_t_side_elts + 1):
        tm[mprime] = 1.j / float(mprime)
        tm[-mprime] = -tm[mprime]

    # From which we can compute the gamma vector:
    Gamma = np.empty(sim.n_xi_tot_elts, dtype=complex)
    for a in range(-sim.n_xi_side_elts, sim.n_xi_side_elts + 1):
        temp_sum = 0. + 0.j
        for mprime in range(max(-sim.n_t_side_elts, a - sim.n_e_side_modes),
                            min(sim.n_t_side_elts, a + sim.n_e_side_modes) + 1):
            temp_sum += tm[mprime] * self.e_fs[a - mprime]
        Gamma[a] = temp_sum / sim.hbar

    # First parentheses counts inversion unknown mode elts
    # Second parentheses counts polarization unknown mode elts
    neqns = (1 + 2 * sim.n_side_modes) + (2 * sim.n_tot_modes)
    # Matrix column layout:
    # N0_re  N_1_re ... N_nsm_re  N_1_im ... N_nsm_im  P_-nsm_re ... P_nsm_re  P_-nsm_im ... P_nsm_im

    # Relevant column start positions in the matrix:
    nm_re_start = 0                               # Start with the real parts of n
    nm_im_start = nm_re_start + sim.n_side_modes  # NOTE: nm_im starts 1 over, but we start at a shift of m = 1
    pm_re_start = nm_im_start + 1 + 2 * sim.n_side_modes  # nsm further PLUS nsm to centre of pm_re elts
                                                          # (which are -/+ offset)
    pm_im_start = pm_re_start + sim.n_tot_modes   # n_tot_modes over from the im_start

    # CONSIDER MIGRATING TO SPARSE MATRIX MODULE
    mat = np.empty([neqns, neqns], dtype=float)
    rhs = np.empty(neqns, dtype=float)

    for p in range(-sim.nsch, sim.nsch + 1):

        # Clear the matrix
        mat.fill(0.)

        # ASSEMBLE

        # First equation
        eqn = nm_re_start
        mat[eqn, eqn] = sim.domega + np.pi / sim.t1
        for mbar in range(1, sim.nint + 1):
            col = nm_im_start + mbar
            mat[eqn, col] = -2. / (sim.t1 * float(mbar))
        for mbar in range(-sim.nint, sim.nint + 1):
            mcheck = mbar + p
            if abs(mcheck) <= sim.n_xi_side_elts:
                col = pm_re_start + mbar
                mat[eqn, col] = 2. * np.imag(Gamma[mcheck])
                col = pm_im_start + mbar
                mat[eqn, col] = 2. * np.real(Gamma[mcheck])
        rhs[eqn] = sim.domega * sim.inv_init_re + np.pi * np.real(sim.lam_inv_fs[0])
        for mprime in range(1, sim.n_side_modes + 1):
            rhs[eqn] -= (2./float(mprime)) * np.imag(sim.lam_inv_fs[mprime])

        # N off-res real equations
        for m in range(1, sim.n_side_modes + 1):
            eqn = nm_re_start + m
            mat[eqn, eqn] = float(m) * sim.domega
            mat[eqn, nm_im_start + m] = 1. / sim.t1
            for mbar in range(-sim.nint, sim.nint + 1):
                mcheck = mbar - m + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start+mbar] -= np.real(self.e_fs[mcheck]) / sim.hbar
                    mat[eqn, pm_im_start+mbar] += np.imag(self.e_fs[mcheck]) / sim.hbar
                mcheck = mbar + m + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start + mbar] += np.real(self.e_fs[mcheck]) / sim.hbar
                    mat[eqn, pm_im_start + mbar] -= np.imag(self.e_fs[mcheck]) / sim.hbar
            rhs[eqn] = np.imag(sim.lam_inv_fs[m]) - np.imag(sim.lam_inv_fs[0])

        # N off-res imag equations
        for m in range(1, sim.n_side_modes + 1):
            eqn = nm_im_start + m
            mat[eqn, eqn] = float(m) * sim.domega
            mat[eqn, nm_re_start + m] = -1. / sim.t1
            mat[eqn, nm_re_start] = 1. / sim.t1
            for mbar in range(-sim.nint, sim.nint + 1):
                mcheck = mbar - m + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start+mbar] -= np.imag(self.e_fs[mcheck]) / sim.hbar
                    mat[eqn, pm_im_start+mbar] -= np.real(self.e_fs[mcheck]) / sim.hbar
                mcheck = mbar + m + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start + mbar] -= np.imag(self.e_fs[mcheck]) / sim.hbar
                    mat[eqn, pm_im_start + mbar] -= np.real(self.e_fs[mcheck]) / sim.hbar
                mcheck = mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start + mbar] += np.imag(self.e_fs[mcheck]) * 2. / sim.hbar
                    mat[eqn, pm_im_start + mbar] += np.real(self.e_fs[mcheck]) * 2. / sim.hbar
            rhs[eqn] = np.real(sim.lam_inv_fs[0]) - np.real(sim.lam_inv_fs[m])

        # P on-res re equations
        eqn = pm_re_start
        mat[eqn, eqn] = sim.domega + np.pi / sim.t2
        mat[eqn, nm_re_start] = -dipfloat * np.imag(Gamma[p])
        for mbar in nonzero_range_trunc:
            mat[eqn, pm_im_start + mbar] = -1. / (float(mbar) * sim.t2)
        for mbar in range(1, sim.nint + 1):
            mcheck = p - mbar
            if abs(mcheck) <= sim.n_xi_side_elts:
                mat[eqn, nm_re_start + mbar] -= dipfloat * np.imag(Gamma[mcheck])
                mat[eqn, nm_im_start + mbar] += dipfloat * np.real(Gamma[mcheck])
            mcheck = p + mbar
            if abs(mcheck) <= sim.n_xi_side_elts:
                mat[eqn, nm_re_start + mbar] -= dipfloat * np.imag(Gamma[mcheck])
                mat[eqn, nm_im_start + mbar] -= dipfloat * np.real(Gamma[mcheck])
        rhs[eqn] = sim.domega * np.real(sim.pol_init) + \
                   np.pi * np.real(sim.lam_pol_fs[0] * sim.v_phase_mults[p])
        for mprime in nonzero_range:
            rhs[eqn] -= np.imag(sim.lam_pol_fs[mprime] * sim.v_phase_mults[p]) / float(mprime)

        # P on-res im equations
        eqn = pm_im_start
        mat[eqn, eqn] = sim.domega + np.pi / sim.t2
        mat[eqn, nm_re_start] = -dipfloat * np.real(Gamma[p])
        for mbar in nonzero_range_trunc:
            mat[eqn, pm_re_start+mbar] = 1. / (float(mbar) * sim.t2)
        for mbar in range(1, sim.nint + 1):
            mcheck = p - mbar
            if abs(mcheck) <= sim.n_xi_side_elts:
                mat[eqn, nm_re_start + mbar] -= dipfloat * np.real(Gamma[mcheck])
                mat[eqn, nm_im_start + mbar] -= dipfloat * np.imag(Gamma[mcheck])
            mcheck = p + mbar
            if abs(mcheck) <= sim.n_xi_side_elts:
                mat[eqn, nm_re_start + mbar] -= dipfloat * np.real(Gamma[mcheck])
                mat[eqn, nm_im_start + mbar] += dipfloat * np.imag(Gamma[mcheck])
        rhs[eqn] = sim.domega * np.imag(sim.pol_init) + \
                   np.pi * np.imag(sim.lam_pol_fs[0] * sim.v_phase_mults[p])
        for mprime in nonzero_range:
            rhs[eqn] += np.real(sim.lam_pol_fs[mprime] * sim.v_phase_mults[p]) / float(mprime)

        # P off-res re equations
        for m in nonzero_range:
            eqn = pm_re_start + m
            mat[eqn, eqn] = sim.domega * float(m)
            mat[eqn, pm_im_start] = -1. / sim.t2
            mat[eqn, pm_im_start + m] = 1. / sim.t2
            mat[eqn, nm_re_start] = np.real(self.e_fs[p]) * dipfloat / sim.hbar
            mcheck = m + p
            if abs(mcheck) <= sim.n_e_side_modes:
                mat[eqn, nm_re_start] -= np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
            for mbar in range(1, sim.nint + 1):
                mcheck = p - mbar
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] -= np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = m - mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] -= np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = m + mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
            rhs[eqn] = np.imag(sim.lam_pol_fs[m]) - np.imag(sim.lam_pol_fs[0] * sim.v_phase_mults[p])

        # P off-res im equations
        for m in nonzero_range:
            eqn = pm_im_start + m
            mat[eqn, eqn] = sim.domega * float(m)
            mat[eqn, pm_re_start] = 1. / sim.t2
            mat[eqn, pm_re_start + m] = -1. / sim.t2
            mat[eqn, nm_re_start] = -np.imag(self.e_fs[p]) * dipfloat / sim.hbar
            mcheck = m + p
            if abs(mcheck) <= sim.n_e_side_modes:
                mat[eqn, nm_re_start] += np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
            for mbar in range(1, sim.nint + 1):
                mcheck = p - mbar
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] += np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = m + mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] += np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
                mcheck = m - mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat / sim.hbar
                    mat[eqn, nm_im_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat / sim.hbar
            rhs[eqn] = np.real(sim.lam_pol_fs[0] * sim.v_phase_mults[p] - sim.lam_pol_fs[m])

        # Solve the system:
        soln = np.linalg.solve(mat, rhs)

        # Distribute the solution:
        self.inv_fs[p, 0] = soln[nm_re_start] + 0.j
        self.inv_fs[p, 1:sim.n_side_modes + 1] = soln[nm_re_start + 1:nm_im_start + 1] \
                                                 + 1.j * soln[nm_im_start + 1:pm_re_start - sim.n_side_modes]
        self.pol_fs[p, 0:sim.n_side_modes + 1] = soln[pm_re_start:pm_re_start + sim.n_side_modes + 1] \
                                                 + 1.j * soln[pm_im_start:pm_im_start + sim.n_side_modes + 1]
        self.pol_fs[p, sim.n_side_modes + 1:sim.n_tot_modes] = soln[pm_re_start - sim.n_side_modes:pm_re_start] \
                                                               + 1.j * soln[pm_im_start-sim.n_side_modes:pm_im_start]


def afzs_rkz_term(afzs_atdpt, afzs_prev):

    # Pull out some names and useful values
    sim = afzs_prev.sim
    temp_c = 1.j * sim.dz * sim.domega / (2. * sim.eps0)

    afzs_rkz = SRwVelFD_Classes.AFZSlice(sim)

    temp_modes = np.zeros(sim.n_e_tot_modes, dtype=complex)

    for p in range(-sim.nsch, sim.nsch + 1):
        for mbar in range(-sim.n_side_modes, sim.n_side_modes+1):
            temp_modes[mbar+p] += sim.fv[p] * afzs_atdpt.pol_fs[p, mbar]

    afzs_rkz.e_fs = afzs_prev.e_fs + temp_c * np.conjugate(temp_modes)

    afzs_rkz.compute_inv_pol_fs_from_e_fs()

    return afzs_rkz


def z_advance(self):
    # Advance the atom field z slice forward one step in z

    # RK1
    afzs_atdpt = self
    afzs_rk1 = afzs_rkz_term(afzs_atdpt, self)

    # RK2
    afzs_atdpt = 0.5 * (self + afzs_rk1)
    afzs_rk2 = afzs_rkz_term(afzs_atdpt, self)

    # RK3
    afzs_atdpt = 0.5 * (self + afzs_rk2)
    afzs_rk3 = afzs_rkz_term(afzs_atdpt, self)

    # RK4
    afzs_atdpt = afzs_rk3
    afzs_rk4 = afzs_rkz_term(afzs_atdpt, self)

    # Assemble RK results
    return (1./6.) * (afzs_rk1 + afzs_rk4) + (1./3.) * (afzs_rk2 + afzs_rk3)


def compute_tseries(self):
    # Computes the inverse discrete fourier transform

    # Grab a name:
    sim = self.sim

    # Inversion tft will need filling out of conjugate modes
    inv_tft_full = np.empty([sim.n_tot_chs, sim.n_tot_modes], dtype=complex)
    for p in range(0, sim.n_tot_chs):
        # Fill out the inversion tft
        inv_tft_full[p, 0] = self.inv_fs[p, 0]
        for m in range(1, sim.n_side_modes + 1):
            inv_tft_full[p, m] = self.inv_fs[p, m]
            inv_tft_full[p, -m] = np.conjugate(self.inv_fs[p, m])
        self.inv_tseries[p, :] = np.real(inverse_trunc_ftx(inv_tft_full[p, :], sim.nt, sim.t_dur))
        self.pol_tseries[p, :] = inverse_trunc_ftx(self.pol_fs[p, :], sim.nt, sim.t_dur)

    # Recall that e is defined to rotate backwards, so need to flip modes:
    e_tft_flipped = np.empty(sim.n_e_tot_modes, dtype=complex)
    e_tft_flipped[0] = self.e_fs[0]
    for m in range(1, sim.n_e_side_modes + 1):
        e_tft_flipped[m] = self.e_fs[-m]
        e_tft_flipped[-m] = self.e_fs[m]
    self.e_tseries = inverse_trunc_ftx(e_tft_flipped, sim.nt, sim.t_dur)


def trunc_ftx(time_series, n_side_modes):
    # The "truncated Fourier transform" is the equivalent of the integral
    # (1/T)*int_0^T{f(t) exp(-i 2pi m t / T)}, which yields the
    # coefficients f_m in a series approximation of f(t)
    # of the form f(t) ~= sum_{m=-nsm}^{n=+nsm} f_m exp(i 2pi mt/T)
    # NOTE: THIS IS A TRUNCATED VERSION OF THE DFT TO SUIT FOURIER SERIES DEFINITION OF PAPER
    #   however, it can be computed from the DFT via this function, which simply drops extra terms
    #   below (derived in my notes, 11 August 2020)
    # NOTE: I invented this terminology. The term "Truncated FT"
    #   seems to exist in the literature; I am unsure if it has
    #   anything to do with this function. See my notes for definitions.

    n_tpts = time_series.shape[0]
    dft = np.fft.fft(time_series)
    fm = np.empty(2 * n_side_modes + 1, dtype=complex)
    for m in range(-n_side_modes, n_side_modes + 1):
        fm[m] = (1. / float(n_tpts)) * dft[m]

    return fm


def inverse_trunc_ftx(tft, nt, t_dur):
    # Compute an inverse truncated fourier transform
    # back to a number of desired points nt

    # tft assumed to have odd number of points;
    # i.e., nsm side modes
    n_modes = tft.shape[0]
    if (n_modes % 2) == 0:
        print("ERROR: ATTEMPTING TO INVERT A TFT ARRAY WITH AN EVEN NUMBER OF ELEMENTS")
        raise ValueError

    n_side_modes = int((n_modes - 1) / 2)
    t_series = np.zeros(nt, dtype=complex)
    for m in range(-n_side_modes, n_side_modes + 1):
        for (ind, t) in enumerate(np.linspace(0.0, t_dur, num=nt, endpoint=True)):
            t_series[ind] += tft[m] * np.exp(1j * 2.0 * np.pi * float(m) * t / t_dur)

    return t_series


def smooth_via_dft_trunc(t_series, nmodes, centre):
    # Smooth a time series within nmodes bandwidth of a central frequency
    #   by taking its DFT and truncating to only adjacent modes
    nt = len(t_series)

    dft_for_trunc = np.roll(np.fft.fft(t_series), -centre)
    dft_for_trunc[nmodes + 1: nt - nmodes].fill(0. + 0.j)
    t_series_clean = np.fft.ifft(dft_for_trunc)

    return t_series_clean
