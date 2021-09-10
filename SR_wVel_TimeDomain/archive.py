# ~~~~~~~~~ ARCHIVE FILE FOR STORING TEMP CODE ~~~~~~~~~~~~~

# CODE SNIPPET FROM 8 SEPTEMBER 2021
# The code below was in operation below the "run" for loop in main.py
#  as of 8 September 2021; it was in use for some trial simulations
#  related to stability across wide velocity distributions.

from scipy.optimize import curve_fit

e_endfire = e_field_trans_avg[params.n_plt_posns-1, :]
int_endfire = np.multiply(np.conj(e_endfire), e_endfire).real
times = np.linspace(0., sim.t_dur_frac, sim.nt_frac)
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Intensity (arb)', color='tab:blue')
ax1.plot(times, inv_trans_avg[params.n_plt_posns-1, 0, :]) #int_endfire)
pump_plot = np.empty(sim.nt_frac, dtype=float)
for (ind, t) in enumerate(times):
    sim.gam_eval(t)
    pump_plot[ind] = sim.gam_npr_cur + \
                     sim.gam_inv_noise * (1.-2.*np.random.rand())
ax2 = ax1.twinx()
ax2.set_ylabel('Pump rate', color='tab:red')
# ax2.set_ylim(bottom=0., top=.5*n0/t1+gam_inv_noise)
ax2.plot(times, pump_plot, color='tab:red')
plt.title('Total vel ch {:1}; total vel width {:5.1e} cm/s; total atoms  {:5.1e}'.format(
    sim.nch, sim.v_width, sim.natoms), y=1.05)
plt.show()
plt.cla()

# Auto-correlate the electric field and fit and plot
e_autocorr = auto_corr_viafft(e_endfire, normalize=False)
e_autocorr_sq = np.multiply(e_autocorr, np.conj(e_autocorr)).real
nac = e_autocorr_sq.shape[0]
tshifts = np.linspace(-sim.t_dur_frac, sim.t_dur_frac, num=nac)

# Perform the fit
[ac_amp, ac_os, ac_wid], pcov = curve_fit(gauss, tshifts, e_autocorr_sq,
                                          p0=[.5*e_autocorr_sq[int(nac/2)], 0., .1 * sim.t_dur_frac])

# Plot
plt.cla()
plt.plot(tshifts, e_autocorr_sq)
# plt.plot(tshifts, gauss(tshifts, ac_amp, ac_os, ac_wid))
plt.title('Electric field auto-correlation squared.')
plt.xlabel('Time shift (s)')
plt.show()

print('The gaussian fit to the squared field auto-correlation has width {:.2e}'.format(ac_wid)+'.')

# # Now repeat the above within different bandwidths of the signal
# coh_posns = range(-sim.nsch+int(coh_bw/2), sim.nsch-int(coh_bw/2)+1)
# coherences = np.empty(len(coh_posns), dtype=float)
#
# for (index, coh_posn) in enumerate(coh_posns):
#     e_rot = rot_view(e_endfire, coh_posn, coh_bw)
#     plt.cla()
#     plt.plot(e_rot.real)
#     plt.plot(e_rot.imag)
#     plt.show()
#     e_autocorr = CorrFns.auto_corr_viafft(e_rot, normalize=False)
#     e_autocorr_sq = np.multiply(e_autocorr, np.conj(e_autocorr)).real
#     [ac_amp, ac_os, ac_wid], pcov = curve_fit(gauss, tshifts, e_autocorr_sq,
#                                               p0=[.5*e_autocorr_sq[int(nac/2)], 0., .1 * sim.t_dur_frac])
#     # [ac_amp, ac_wid], pcov = curve_fit(lorentz, tshifts, e_autocorr_sq,
#     #                                     p0=[.5*e_autocorr_sq[int(nac/2)], .4 * sim.t_dur_frac])
#     ac_wid = sim.dt * find_width(e_autocorr_sq)
#     coherences[index] = ac_wid
#
#     # Plot
#     plt.cla()
#     plt.plot(tshifts, e_autocorr_sq)
#     plt.plot(tshifts, gauss(tshifts, ac_amp, 0., ac_wid))
#     # plt.plot(tshifts, lorentz(tshifts, ac_amp, ac_wid))
#     plt.title('E ac sq, shift {shift}, fit wid {wid:.2e}'.format(shift=coh_posn, wid=ac_wid))
#     plt.xlabel('Time shift (s)')
#     plt.show()
#
# plt.cla()
# plt.plot(coh_posns, coherences)
# plt.show()

# for ind in range(0, n_plt_posns):
#     autocorr = CorrFns.auto_corr_viafft(e_field_trans_avg[ind,:])
#     plt.plot(np.real(autocorr), linewidth=.5)
#     plt.show()

# Pickle the data for plotting outside of this python script
# fname = 'td_intensities_nch' + str(sim.nch) + '.pickle'
# with open(fname, 'wb') as f:
#     pickle.dump(int_trans_avg, f)

fname = 'td_e_endfire_nch' + str(sim.nch) + '.pickle'
with open(fname, 'wb') as f:
    pickle.dump(e_endfire, f)

# fname = 'td_inversions_nch' + str(sim.nch) + '.pickle'
# with open(fname, 'wb') as f:
#     pickle.dump(inv_trans_avg, f)
#
# fname = 'td_polarisations_nch' + str(sim.nch) + '.pickle'
# with open(fname, 'wb') as f:
#     pickle.dump(p_trans_avg, f)


total_energy = sim.dt * np.sum(int_endfire)
energy_string = "Total energy (arb units, see code) {:.2e}".format(total_energy)

# Auto-correlate the intensity
int_autocorr = auto_corr_viafft(int_endfire, normalize=False).real
int_autocorr_smoothed = crop_fs(int_autocorr, 0, params.smoothing_bw).real

# Perform the curve fitting
int_max_estimate = np.amax(int_autocorr_smoothed)
int_wid_estimate = sim.nt_frac / 5
nac = int_autocorr.shape[0]
xdata = np.array(range(-int(nac/2), int(nac/2)+1))
[ac_amp, ac_os, ac_wid], pcov = curve_fit(gauss, xdata, int_autocorr, p0=[int_max_estimate, 0, int_wid_estimate])


plt.cla()
plt.plot(xdata, int_autocorr)
plt.plot(xdata, gauss(xdata, ac_amp, ac_os, ac_wid))
plt.title('Intensity auto-correlation')
plt.show()

# Peak intensity value (arb units)  and time of occurrence
peak_int = np.sqrt(ac_amp/sim.nt_frac)
peak_time = float(np.argmax(int_endfire)) * sim.dt
pulse_wid = float(ac_wid * sim.dt)
intensity_string = "Peak intensity (arb units, see code) of {:.2e}".format(peak_int) +\
                   " @ t = {:.2e}".format(peak_time) +\
                   " of width {:.2e}".format(pulse_wid)
print(intensity_string)
print(energy_string)

# Next, the width. We use a mean square deviation metric:
# tvec = np.linspace(0, t_dur, num=sim.nt_frac, endpoint=True)
# int_sum = np.sum(int_trans_avg[sim.n_plt_posns-1, :])
# mean_t = np.dot(int_trans_avg[sim.n_plt_posns-1, :], tvec) / int_sum
# dev_vec = tvec - mean_t
# del_t = (1./np.sqrt(int_sum)) * np.sqrt(np.dot(int_trans_avg[sim.n_plt_posns-1, :], np.multiply(dev_vec, dev_vec)))
# print("Pulse width of {:.2e}".format(del_t) + " s.")

# # Save the spectrum:
# x = np.linspace(0, t_dur, num=sim.nt_frac, endpoint=True)
# plt.cla()
# p_spectra = np.empty((sim.nch, sim.nt_frac), dtype=float)
# for k in range(0, sim.nsch + 1):
#     p_spectra[k+sim.nsch, :] = CorrFns.power_spectrum(p_trans_avg[-1, k, :])
# for k in range(sim.nsch + 1, sim.nch):
#     p_spectra[k - sim.nsch - 1, :] = CorrFns.power_spectrum(p_trans_avg[-1, k, :])
# plt.imshow(np.transpose(p_spectra))
# plt.savefig("SpectralRaster", dpi=160)
# plt.cla()

simtime_end = time.time()

print("Total execution time " + str(simtime_end-simtime_start) + " s.")