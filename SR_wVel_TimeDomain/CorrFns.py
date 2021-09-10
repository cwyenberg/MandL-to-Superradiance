import numpy as np


def auto_corr(data, normalize=True):

    # Performs an autocorrelation on data
    # Note that for data size n_data, we autocorrelate to 2*n_data-1 points
    # (i.e. this returns a larger array than passed)

    n_data = data.shape[0]          # Number of data points in the input data
    n_ac = 2*n_data-1               # Autocorrelation array size
    result = np.empty(n_ac)         # Allocate the result

    # Compute the result
    for index in range(0, n_ac):
        data_lo = max(index+1-n_data,0)
        data_hi = min(index+1, n_data)
        conj_lo = max(n_data-1-index,0)
        conj_hi = min(2*n_data-1-index, n_data)
        result[index] = np.vdot(data[conj_lo:conj_hi], data[data_lo:data_hi])

    if normalize:
        return result/n_data
    else:
        return result


def auto_corr2D_viafft(data, normalize=True):

    # Returns a 2D autocorrelation computed via an intermediate FFT

    # Number of data pts
    nx, ny = data.shape[0], data.shape[1]

    padded = np.append(data, np.zeros((nx,ny)), axis=0)
    padded = np.append(padded, np.zeros((2*nx,ny)), axis=1)

    # Perform the FFT
    data_dft = np.fft.fft2(padded)

    # DFT of auto-correlation is simply (conjugate) multiplication
    # Elt-wise multiplication of fft
    data_ac_dft = np.multiply(np.conjugate(data_dft), data_dft)

    # Inverse FFT to return to time
    # Note this array will be half-shifted
    result_shifted = np.fft.ifft2(data_ac_dft)

    # Flip the result array around
    return_shape = (result_shifted.shape[0]-1, result_shifted.shape[1]-1)
    temp_array_a = np.empty(return_shape)
    temp_array_b = np.empty(return_shape)

    # Flip in x:
    temp_array_a[0:nx,:] = result_shifted[nx-1:2*nx-1,0:2*ny-1]
    temp_array_a[nx:2*nx-1,:] = result_shifted[0:nx-1,0:2*ny-1]
    # Flip in y:
    temp_array_b[:,0:ny] = temp_array_a[:,ny-1:2*ny-1]
    temp_array_b[:,ny:2*ny-1] = temp_array_a[:,0:ny-1]

    if normalize:
        return temp_array_b/float(nx*ny)
    else:
        return temp_array_b


def auto_corr_viafft(data, normalize=True):

    # Returns an autocorrelation computed via an intermediate FFT

    # Number of data pts
    nt = data.shape[0]

    # Perform the FFT
    data_dft = np.fft.fft(np.append(data, np.zeros(nt)))

    # DFT of auto-correlation is simply (conjugate) multiplication
    # Elt-wise multiplication of fft
    data_ac_dft = np.multiply(np.conjugate(data_dft), data_dft)

    # Inverse FFT to return to time
    # Note this array will be half-shifted
    result_shifted = np.fft.ifft(data_ac_dft)

    # Allocate result
    result = np.empty(2*nt-1, dtype=complex)

    # Flip the result
    result[0:nt] = result_shifted[nt-1:2*nt-1]
    result[nt:2*nt-1] = result_shifted[0:nt-1]

    # Return result
    if normalize:
        return result/float(nt)
    else:
        return result


def power_spectrum(data, normalize=True):

    # Returns the power spectrum

    # Number of data pts
    nt = data.shape[0]

    # Perform the fft
    data_dft = np.fft.fft(data)

    # Power spectrum is elt-wise norm of fft components
    spec = np.multiply(np.conjugate(data_dft), data_dft)

    result = np.empty(nt, dtype=float)
    result[0:int(nt/2)] = np.real(spec[nt-int(nt/2):nt])
    result[int(nt/2):nt] = np.real(spec[0:nt-int(nt/2)])

    # Return result
    if normalize:
        return result/float(nt)
    else:
        return result


def gauss(x, amp, xos, wid):
    return amp * np.exp(-((x-xos)/wid)**2)


def lorentz(x, amp, wid):
    return amp * wid / (x*x + wid*wid)
