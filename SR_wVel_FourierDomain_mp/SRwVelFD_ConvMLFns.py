# CONVENTIONAL MENEGOZZI AND LAMB FUNCTIONS

# Imports
import numpy as np


def compute_inv_pol_fs_from_e_fs_conv_ml(self):
    # Solves for inversion and polarisation Fourier series
    # from given electric field Fourier series
    # USES CONVENTIONAL MENEGOZZI AND LAMB THEORY

    # Mode:
    #       1: Constrain initial conditions, drop m=0 eqn           (AKA TRANSIENT ML)
    #       2: Don't constrain initial conditions, keep m=0 eqn     (AKA NORMAL ML)
    #       3: Constrain initial conditions, keep m=0 eqn,
    #          augment with inv/pol dirac pumps of unknown amplitude
    #          (THIS METHOD DOES NOT WORK VERY WELL, NOT IN PAPER)
    #       4: Constrain initial conditions, overconstrain
    #          the system, and solve the normal equations A^T A x = A^T b
    #          (THIS METHOD FAILED IN TESTING, NOT IN PAPER)
    mode = 1

    # Simply name the current mode for legibility below...
    FIC_drop_meq0 = (mode == 1)
    No_FIC = (mode == 2)
    FIC_aug_pumps = (mode == 3)
    FIC_NormalEqns = (mode == 4)

    # Pull out the simulation data name for convenience
    sim = self.sim

    dipfloat = 2. * sim.dip * sim.dip / sim.hbar   # A useful quantity

    # First parentheses counts inversion unknown mode elts
    # Second parentheses counts polarization unknown mode elts
    neqns = (1 + 2 * sim.n_side_modes) + (2 * sim.n_tot_modes)

    # Add 3x unknowns for fiducial pumps (1x inv re, 2x pol re+im) if in that mode:
    if FIC_aug_pumps:
        neqns += 3
        alpha_re_index = neqns - 3
        FIC_inv_re_eqn = alpha_re_index
        beta_re_index = neqns - 2
        FIC_pol_re_eqn = beta_re_index
        beta_im_index = neqns - 1
        FIC_pol_im_eqn = beta_im_index
        ncols = neqns

    # Facilitate normal equations if in that mode:
    if FIC_NormalEqns:
        ncols = neqns
        neqns += 3
        FIC_inv_re_eqn = neqns - 3
        FIC_pol_re_eqn = neqns - 2
        FIC_pol_im_eqn = neqns - 1

    # Adjust to force initial conditions if in either of those two modes:
    if FIC_drop_meq0 or No_FIC:
        ncols = neqns

    # Matrix column layout:
    # N0_re  N_1_re ... N_nsm_re  N_1_im ... N_nsm_im  P_-nsm_re ... P_nsm_re  P_-nsm_im ... P_nsm_im
    # Last 2x columns reserved for fiducial pump coefficients if enabled

    # Relevant column start positions in the matrix:
    nm_re_start = 0                               # Start with the real parts of n
    nm_im_start = nm_re_start + sim.n_side_modes  # NOTE: nm_im starts 1 over, but we start at a shift of m = 1
    pm_re_start = nm_im_start + 1 + 2 * sim.n_side_modes  # nsm further PLUS nsm to centre of pm_re elts
                                                          # (which are -/+ offset)
    pm_im_start = pm_re_start + sim.n_tot_modes   # n_tot_modes over from the im_start

    # CONSIDER MIGRATING TO SPARSE MATRIX MODULE
    mat = np.empty([neqns, ncols], dtype=float)
    rhs = np.empty(neqns, dtype=float)

    # Solve for all velocity channels
    for p in range(-sim.nsch, sim.nsch + 1):

        # Clear the matrix
        mat.fill(0.)

        # ASSEMBLE

        # INV EQUATIONS:

        # First two equations are the equations of constraint
        # for real and imaginary initial conditions on N (if enabled);
        # the latter is a non-existent constraint, see note below.

        # IC constraints on Re{N} replace m=0 eqn in this mode:
        if FIC_drop_meq0:
            eqn = nm_re_start
            mat[eqn, eqn] = 1.
            for mbar in range(1, sim.n_side_modes + 1):
                col = nm_re_start + mbar
                mat[eqn, col] = 2.
            rhs[eqn] = sim.inv_init_re
            # !!! IC constraints on Im{N} do not need to be established by explicit relations;
            # they are already implicitly enforced by our assumption that N is real (and our exploitation
            # of that fact in our derivation of the below equations)!!!

        # Real parts of equations for N
        for m in range(0, sim.n_side_modes + 1):
            if (m == 0) and FIC_drop_meq0:
                continue
            eqn = nm_re_start + m
            col = nm_im_start + m
            mat[eqn, col] = float(m) * sim.domega
            mat[eqn, nm_re_start + m] = -1. / sim.t1
            for mbar in range(-sim.nint, sim.nint + 1):
                mcheck = mbar - m + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, pm_re_start + mbar] -= (2. / sim.hbar) * np.imag(self.e_fs[mcheck])
                    mat[eqn, pm_im_start + mbar] -= (2. / sim.hbar) * np.real(self.e_fs[mcheck])
            rhs[eqn] = -np.real(sim.lam_inv_fs[m])

            # Add the augmented pump unknown:
            if FIC_aug_pumps:
                mat[eqn, alpha_re_index] = 1.

        # Off-res imaginary parts of equations for N
        for m in range(1, sim.n_side_modes + 1):
            eqn = nm_im_start + m
            col = nm_re_start + m
            mat[eqn, col] = float(m) * sim.domega
            mat[eqn, nm_im_start + m] = 1. / sim.t1
            rhs[eqn] = np.imag(sim.lam_inv_fs[m])

        # POL EQUATIONS:

        # First two equations are the equations of constraint
        # for real and imaginary initial conditions on P (if enabled):

        if FIC_drop_meq0:
            # IC constraints on Re{P}:
            eqn = pm_re_start
            for mbar in range(-sim.n_side_modes, sim.n_side_modes + 1):
                col = pm_re_start + mbar
                mat[eqn, col] = 1.
            rhs[eqn] = np.real(sim.lam_pol_fs[0] * sim.v_phase_mults[p])
            # IC constraints on Im{P}:
            eqn = pm_im_start
            for mbar in range(-sim.n_side_modes, sim.n_side_modes + 1):
                col = pm_im_start + mbar
                mat[eqn, col] = 1.
            rhs[eqn] = np.imag(sim.lam_pol_fs[0] * sim.v_phase_mults[p])

        # Real parts of equations for P
        for m in range(-sim.n_side_modes, sim.n_side_modes + 1):
            if (m == 0) and FIC_drop_meq0:
                continue
            eqn = pm_re_start + m
            col = pm_im_start + m
            mat[eqn, col] = sim.domega * float(m)
            mat[eqn, pm_re_start + m] = -1. / sim.t2
            mcheck = m + p
            if abs(mcheck) <= sim.n_e_side_modes:
                mat[eqn, nm_re_start] += np.imag(self.e_fs[mcheck]) * dipfloat
            for mbar in range(1, sim.nint + 1):
                mcheck = m + mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat
                    mat[eqn, nm_im_start + mbar] += np.real(self.e_fs[mcheck]) * dipfloat
                mcheck = m - mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat
                    mat[eqn, nm_im_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat
            rhs[eqn] = -np.real(sim.lam_pol_fs[m])

            # Add the augmented pump unknown:
            if FIC_aug_pumps:
                mat[eqn, beta_re_index] = 1.

        # Imag parts of equations for P
        for m in range(-sim.n_side_modes, sim.n_side_modes + 1):
            if (m == 0) and FIC_drop_meq0:
                continue
            eqn = pm_im_start + m
            col = pm_re_start + m
            mat[eqn, col] = sim.domega * float(m)
            mat[eqn, pm_im_start + m] = 1. / sim.t2
            mcheck = m + p
            if abs(mcheck) <= sim.n_e_side_modes:
                mat[eqn, nm_re_start] -= np.real(self.e_fs[mcheck]) * dipfloat
            for mbar in range(1, sim.nint + 1):
                mcheck = m - mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat
                    mat[eqn, nm_im_start + mbar] -= np.imag(self.e_fs[mcheck]) * dipfloat
                mcheck = m + mbar + p
                if abs(mcheck) <= sim.n_e_side_modes:
                    mat[eqn, nm_re_start + mbar] -= np.real(self.e_fs[mcheck]) * dipfloat
                    mat[eqn, nm_im_start + mbar] += np.imag(self.e_fs[mcheck]) * dipfloat
            rhs[eqn] = np.imag(sim.lam_pol_fs[m])

            # Add the augmented pump unknown:
            if FIC_aug_pumps:
                mat[eqn, beta_im_index] = -1.

        # Add the initial condition constraints at the end if in FIC_aug_pumps or FIC_NormalEqns modes
        if FIC_aug_pumps or FIC_NormalEqns:
            # IC constraints on Re{N}:
            mat[FIC_inv_re_eqn, nm_re_start] = 1.
            for mbar in range(1, sim.n_side_modes + 1):
                col = nm_re_start + mbar
                mat[FIC_inv_re_eqn, col] = 2.
            rhs[FIC_inv_re_eqn] = sim.inv_init_re
            # !!! IC constraints on Im{N} do not need to be established by explicit relations;
            # they are already implicitly enforced by our assumption that N is real (and our exploitation
            # of that fact in our derivation of the below equations)!!!

            # IC constraints on Re{P}:
            for mbar in range(-sim.n_side_modes, sim.n_side_modes + 1):
                col = pm_re_start + mbar
                mat[FIC_pol_re_eqn, col] = 1.
            rhs[FIC_pol_re_eqn] = np.real(sim.lam_pol_fs[0] * sim.v_phase_mults[p])
            # IC constraints on Im{P}:
            for mbar in range(-sim.n_side_modes, sim.n_side_modes + 1):
                col = pm_im_start + mbar
                mat[FIC_pol_im_eqn, col] = 1.
            rhs[FIC_pol_im_eqn] = np.imag(sim.lam_pol_fs[0] * sim.v_phase_mults[p])

        # Solve the system:
        if not FIC_NormalEqns:
            soln = np.linalg.solve(mat, rhs)
        else:
            mat_t = np.transpose(mat)
            soln = np.linalg.solve(np.matmul(mat_t, mat), np.matmul(mat_t, rhs))

        # Distribute the solution:
        self.inv_fs[p, 0] = soln[nm_re_start] + 0.j
        self.inv_fs[p, 1:sim.n_side_modes + 1] = soln[nm_re_start + 1:nm_im_start + 1] \
                                                 + 1.j * soln[nm_im_start + 1:pm_re_start - sim.n_side_modes]
        self.pol_fs[p, 0:sim.n_side_modes + 1] = soln[pm_re_start:pm_re_start + sim.n_side_modes + 1] \
                                                 + 1.j * soln[pm_im_start:pm_im_start + sim.n_side_modes + 1]
        self.pol_fs[p, sim.n_side_modes + 1:sim.n_tot_modes] = soln[pm_re_start - sim.n_side_modes:pm_re_start] \
                                                               + 1.j * soln[pm_im_start-sim.n_side_modes:pm_im_start]
