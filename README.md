# MandL-to-Superradiance

This code supplements the paper entitled "Generalisation of the Menegozzi & Lamb Maser Algorithm to the Transient Superradiance Regime" by CMW et al.

The three folders correspond to:
* SR_wVel_TimeDomain:  Time domain simulation of the Maxwell-Bloch equations over a velocity distribution
* SR_wVel_FourierDomain_mp:  Fourier domain simulations of the Maxwell-Bloch equations over a velocity distribution (with multiprocessing). This code can execute a conventional Menegozzi and Lamb algorithm in the quasi-steady state domain, a transient Menegozzi and Lamb algorithm convergent in the weak field limit, and the Integral Fourier (IF) algorithm unique to the paper and convergent in the transient regime at all field strengths.
* SR_wVel_Plotting:  Plotting scripts. All object data is pickled after simulation execution in either the time or Fourier domains; pickled data must be placed in the same folder as these Python scripts before their execution to generate the figures of the paper.

Refer to the code comments for more detailed usage. Simulation parameters are input at the top of each ...main.py script. Mode selection in the FourierDomain script involves a few steps (modifying solver method mappings to swap between IF and Menegozzi and Lamb algorithms, and sub-mode index selection when working within the Menegozzi and Lamb algorithm to select quasi-steady state, transient, or overconstrained methods). Refer to code comments for clear instructions.
