# GENERIC (NON RUNGE-KUTTA) FUNCTIONS FOR TD SIMULATION OF MB EQUATIONS

# Import the usual modules
import numpy as np
import SRwVelTD_Classes
import time

# Plotting
import matplotlib.pyplot as plt


def int_comp(self):
    # Compute my intensity profile
    # (across modes AND totalized):
    self.int_tot = np.power(self.ep_re, 2) + np.power(self.ep_im, 2)


def atmfld_lin_comb(sf_a, atmfld_a, sf_b, atmfld_b, sim):
    # This function constructs a linear combination of two input atomic fields

    atmfld_ret = SRwVelTD_Classes.AtomFieldProfile(sim)

    atmfld_ret.nkp_re = sf_a * atmfld_a.nkp_re + sf_b * atmfld_b.nkp_re
    atmfld_ret.nkp_im = sf_a * atmfld_a.nkp_im + sf_b * atmfld_b.nkp_im
    atmfld_ret.pkp_re = sf_a * atmfld_a.pkp_re + sf_b * atmfld_b.pkp_re
    atmfld_ret.pkp_im = sf_a * atmfld_a.pkp_im + sf_b * atmfld_b.pkp_im
    atmfld_ret.ep_re = sf_a * atmfld_a.ep_re + sf_b * atmfld_b.ep_re
    atmfld_ret.ep_im = sf_a * atmfld_a.ep_im + sf_b * atmfld_b.ep_im

    return atmfld_ret


def gaussian(size, wid):

    hsize = int(size/2)
    gaussian = np.exp(-((range(0, size)-hsize)/wid)**2) / (wid * np.sqrt(np.pi))

    return np.roll(gaussian, hsize)


def gauss_filter(arr, wid):
    # Construct the filter
    nt = arr.shape[0]
    gaussian = np.empty(nt, dtype=float)
    for ind in range(-int(nt/2), int(nt/2)):
        gaussian[ind] = np.exp(-(float(ind)/float(wid))**2.) / (wid * np.sqrt(np.pi))

    arr_fft = np.fft.fft(arr)
    gaussian_fft = np.fft.fft(gaussian)
    fft_filtered = np.multiply(np.conj(arr_fft), gaussian_fft)

    arr_filtered = np.fft.ifft(fft_filtered).real

    return np.flip(arr_filtered)


def simulate(sim, animate, plotstep, bandstep):
    # Perform one simulation (of potentially multiple if averaging)

    # Randomize the phases:
    if sim.rand_things:
        sim.randomize()

    # Allocate the Atom and Field Class;
    # Force a propagation upon allocation:
    Atm_Fld = SRwVelTD_Classes.AtomFieldProfile(sim, init=True, force_propz=True)

    # Allocate total intensity transients
    int_tot_transients = np.zeros((sim.n_plt_posns, sim.nt_frac), dtype=float)

    # Compute the starting intensities
    Atm_Fld.int_comp()

    # Plot the electric field
    e_field_transients = np.empty((sim.n_plt_posns, sim.nt_frac), dtype=complex)
    for z_zone in range(0, sim.n_plt_posns):
        z_ind = int((z_zone + 1) * (sim.nz-1) / sim.n_plt_posns)
        e_field_transients[z_zone, 0] = Atm_Fld.ep_re[z_ind] + 1.j * Atm_Fld.ep_im[z_ind]
        int_tot_transients[z_zone, 0] = Atm_Fld.int_tot[z_ind]

    # Allocate and initialize the polarization and inversion trends
    p_transients = np.empty((sim.n_plt_posns, sim.nch, sim.nt_frac), dtype=complex)
    inv_transients = np.empty((sim.n_plt_posns, sim.nch, sim.nt_frac), dtype=float)
    for z_zone in range(0, sim.n_plt_posns):
        z_ind = int((z_zone + 1) * (sim.nz-1) / sim.n_plt_posns)
        inv_transients[z_zone, :, 0] = Atm_Fld.nkp_re[:, z_ind]
        p_transients[z_zone, :, 0] = Atm_Fld.pkp_re[:, z_ind] + Atm_Fld.pkp_im[:, z_ind] * 1j

    # Allocate the pump transient and initialize
    pump_transient = np.empty(sim.nt_frac, dtype=float)
    pump_transient[0] = sim.gam_npr_cur

    # Just a range to store the channels
    chnls = range(0, sim.nch)

    # Execute the simulation
    for t in range(1, sim.nt_frac):

        if t % plotstep == 1 or plotstep == 1:
            t_start = time.time()

        # Perform a time step
        sim.t_cur = float(t) * sim.dt
        sim.gam_eval(sim.t_cur)

        # Plot the pump
        pump_transient[t] = sim.gam_npr_cur
        # Step forward in time
        Atm_Fld.t_step(sim)
        # Compute the intensity for plotting
        Atm_Fld.int_comp()

        # Compile the transients
        # Plot the electric field
        for z_zone in range(0, sim.n_plt_posns):
            z_ind = int((z_zone + 1) * (sim.nz-1) / sim.n_plt_posns)
            e_field_transients[z_zone, t] = Atm_Fld.ep_re[z_ind] + 1.j * Atm_Fld.ep_im[z_ind]
            int_tot_transients[z_zone, t] = Atm_Fld.int_tot[z_ind]
            inv_transients[z_zone, :, t] = Atm_Fld.nkp_re[:, z_ind]
            p_transients[z_zone, :, t] = Atm_Fld.pkp_re[:, z_ind] + Atm_Fld.pkp_im[:, z_ind] * 1j

        # Timing and animation:
        if t % plotstep == 0:

            # Animate the intensity if requested:
            if animate:
                # for p in chnls:
                #     if p % bandstep == 0:
                        # plt.plot(np.multiply(Atm_Fld.ep_re[:],Atm_Fld.ep_re[:])
                        #          + np.multiply(Atm_Fld.ep_im[:],Atm_Fld.ep_im[:]), linewidth=.5)
                        # plt.plot(Atm_Fld.pkp_re[p, :], linewidth=0.5)
                        # plt.plot(np.angle(Atm_Fld.pkp_re[p, :] + 1.j * Atm_Fld.pkp_im[p,:]), linewidth=.5)
                        # plt.plot(Atm_Fld.nkp_re[p, :], linewidth=.5)
                plt.plot(int_tot_transients[sim.n_plt_posns-1, :], linewidth=.5)
                # int_filtered = gauss_filter(int_tot_transients[sim.n_plt_posns-1,:], 50)
                # plt.plot(int_filtered, linewidth=.5)

                plt.title('Intensity thus far')
                plt.xlabel('time (steps)')
                plt.show(block=False)
                plt.pause(0.01)
                plt.cla()

            t_end = time.time()             # Timing
            print("t = " + str(t) + " step completed after " + str(t_end - t_start) + " sec.")

    return int_tot_transients, e_field_transients, p_transients, inv_transients, pump_transient


def highfreq_power_trans(transient, nmodes, ndsample=1):
    # UNUSED IN REF_PAPER
    # Computes the higher-frequency power transients
    # nmodes:   number of modes to compute; i.e., number of adjacent time points to
    #           perform local fft on
    # ndsample: allows the caller to downsample the transient before processing
    #           averages over points over an interval of ndsample

    # Get the time length
    nt = transient.shape[0]

    # Perform the downsampling
    nt_ds = int(nt/ndsample)
    trans_ds = np.empty(nt_ds, dtype=complex)
    for t in range(0, nt_ds):
        trans_ds[t] = np.average(transient[t*ndsample:t*ndsample+ndsample])

    # Construct the fourier array
    trans_modes = np.empty([nmodes, nt_ds], dtype=complex)
    adj_lo = int(nmodes/2)
    adj_hi = nmodes-adj_lo
    for t in range(adj_lo, nt_ds-adj_hi):
        trans_modes[:, t] = np.fft.fft(trans_ds[t-adj_lo:t+adj_hi])
    # Pad in the start and finish:
    for t in range(0, adj_lo):
        trans_modes[:, t] = trans_modes[:, adj_lo]
    for t in range(nt_ds-adj_hi, nt_ds):
        trans_modes[:, t] = trans_modes[:, nt_ds-adj_hi-1]

    # Compute the power transients
    trans_power = np.empty([nmodes, nt_ds], dtype=float)
    for k in range(0, nmodes):
        trans_power[k, :] = np.real(np.multiply(np.conjugate(trans_modes[k, :]), trans_modes[k, :]))

    # Upsample back
    modes_return = np.empty([nmodes, nt], dtype=complex)
    power_return = np.empty([nmodes, nt], dtype=float)
    for tds in range(0, nt_ds):
        for k in range(0, nmodes):
            modes_return[k, ndsample*tds:ndsample*tds+ndsample].fill(trans_modes[k, tds])
            power_return[k, ndsample*tds:ndsample*tds+ndsample].fill(trans_power[k, tds])

    return modes_return, power_return


def crop_fs(array, centre, bw):
    # Crops the Fourier Series about the centre requested
    #   to the bandwidth requested
    hbw = int(bw/2)
    length = array.shape[0]
    array_fs = np.fft.fft(array)
    fs_rolled = np.roll(array_fs, -centre)
    fs_rolled[hbw:length-hbw+1].fill(0.+0.j)
    fs_filtered = np.roll(fs_rolled, centre)

    return np.fft.ifft(fs_filtered)


def rot_view(array, centre, bw):
    # Crops the Fourier Series about the centre requested
    #   to the bandwidth requested
    hbw = int(bw/2)
    length = array.shape[0]
    array_fs = np.fft.fft(array)
    fs_rolled = np.roll(array_fs, -centre)
    fs_rolled[hbw:length-hbw+1].fill(0.+0.j)

    return np.fft.ifft(fs_rolled)


def find_width(array):
    # Find a characteristic width of a pulse
    nx = array.shape[0]
    x_vec = np.linspace(0, nx, num=nx, endpoint=False)
    arr_sum = np.sum(array)
    x_mean = np.dot(array, x_vec) / arr_sum
    dev_vec = np.abs(x_vec - x_mean)
    width = np.dot(dev_vec, array) / arr_sum

    return width
