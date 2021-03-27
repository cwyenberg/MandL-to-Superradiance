import numpy as np
import SRwVelFD_Classes
import multiprocessing as mp

# Some functions within this module have been
# parallelized for multiple processors


def compute_inv_pol_fs_from_e_fs_if_mp(self):
    # Solves for inversion and polarisation Fourier series
    # from given electric field Fourier series
    # USES INTEGRAL FOURIER METHOD WITH MULTIPROCESSING

    # Pull out the simulation  data name for convenience
    sim = self.sim

    # A few useful quantities:
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

    # Arguments for the pooled fn
    ch_solver_args = (self,
                      sim,
                      nonzero_range,
                      nonzero_range_trunc,
                      dipfloat,
                      tm,
                      Gamma,
                      neqns,
                      nm_re_start,
                      nm_im_start,
                      pm_re_start,
                      pm_im_start)

    chs = range(-sim.nsch, sim.nsch + 1)

    if sim.mp_enabled:

        # Allocate the list of arguments
        pool_args = [(p, ch_solver_args) for p in chs]
        # Create the pool
        pool = mp.Pool(mp.cpu_count())
        # Map tasks to the pool
        pool_result = pool.starmap(ch_solver, pool_args)
        # Close the pool
        pool.close()
        # Wait for the pool to finish
        pool.join()
        # Distribute the result
        for index in range(0, sim.n_tot_chs):
            p = index - sim.nsch
            self.inv_fs[p, :], self.pol_fs[p, :] = pool_result[index]

    else:
        for p in chs:
            self.inv_fs[p, :], self.pol_fs[p, :] = ch_solver(p, ch_solver_args)


def ch_solver(p, ch_solver_args):

    # Unpack the args:
    (self,
     sim,
     nonzero_range,
     nonzero_range_trunc,
     dipfloat,
     tm,
     Gamma,
     neqns,
     nm_re_start,
     nm_im_start,
     pm_re_start,
     pm_im_start) = ch_solver_args

    mat = np.zeros([neqns, neqns], dtype=float)
    rhs = np.empty(neqns, dtype=float)

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

    # Distribute the solution

    # Allocate:
    return_inv_tft = np.empty(sim.n_side_modes+1, dtype=complex)
    return_pol_tft = np.empty(sim.n_tot_modes, dtype=complex)

    # Distribute:
    return_inv_tft[0] = soln[nm_re_start] + 0.j
    return_inv_tft[1:sim.n_side_modes + 1] = soln[nm_re_start + 1:nm_im_start + 1] \
                         + 1.j * soln[nm_im_start + 1:pm_re_start - sim.n_side_modes]
    return_pol_tft[0:sim.n_side_modes + 1] = soln[pm_re_start:pm_re_start + sim.n_side_modes + 1] \
                         + 1.j * soln[pm_im_start:pm_im_start + sim.n_side_modes + 1]
    return_pol_tft[sim.n_side_modes + 1:sim.n_tot_modes] = soln[pm_re_start-sim.n_side_modes:pm_re_start] \
                         + 1.j * soln[pm_im_start-sim.n_side_modes:pm_im_start]

    # Return to caller:
    return return_inv_tft, return_pol_tft
