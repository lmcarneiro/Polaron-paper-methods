from scipy.integrate import solve_ivp
import scipy as sc
import scipy.integrate as sci
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import statsmodels.api as sm
from TAanalysis import plotTA
import customplots as cp
import os
import matplotlib.ticker as mticker

def decomposition(TA_file, init_time, fin_time, wav, semilog=False, n=None):
    """First decompose into initial and final state. The fit will be to these
    two states."""

    if '.txt' in TA_file:
        TA = np.loadtxt(TA_file)
    else:
        TA = np.load(TA_file)
    times = TA[1:,0]
    energies = TA[0][1:]
    delA = TA[1:,1:]
    init_state = delA[init_time]
    fin_state = delA[fin_time]

    if n is not None:
    #Easy fix to avoid division by 0 in RLM (due to perfect fit).
    #Could also go back and make it so CollectSPE() doesn't average the SPE
    #files together beforehand, just in this case. This works fine so far tho.
        init_state = np.average(delA[init_time-n:init_time+n],0)
        fin_state = np.average(delA[fin_time-n:fin_time+n],0)
    Y = np.asarray([init_state, fin_state])
    slopeOut_list = []
    stdOut_list = []

    for i, time in enumerate(times):

        model = sm.RLM(delA[i], Y.transpose(), M=sm.robust.norms.TukeyBiweight()).fit()
        slopeOut_list.append(model.params)
        stdOut_list.append(model.bse)

    slopeOut_list = np.asarray(slopeOut_list)
    totals = [(slopeOut_list[i,0]*init_state + slopeOut_list[i,1]*fin_state)/2.
              for i, time in enumerate(times)]

    init_coeff = slopeOut_list[:,0]
    fin_coeff = slopeOut_list[:,1]
    coeffs = np.array([init_coeff, fin_coeff])

    plotTA(energies, times, wav, init=init_state, fin=fin_state, totals=totals,
           contour = True, semilog=semilog)

    np.save('coeffs', coeffs)
    return coeffs, times

def pulse(t, dt, delay, start):
    """Excitation pulse equation. Used by rate()."""

    tspan = np.linspace(start, start+1000, 100) #short pulse, don't need many points
    fun = [1/(np.cosh(-(t-delay)/dt))**2 for t in tspan]
    norm = sci.trapz(fun, tspan)

    return (np.cosh(-(t-delay)/dt)**-2)/norm

def rate(t, y, dt, elph_tau, pol_tau, delay, start):
    """Rate equation function for two state model. y[0] is charge transfer state,
    y[1] is polaron state, elph_tau is electron-phonon scattering constant, pol_tau is
    polaron formation constant."""

    dydt = [(pulse(t, dt, delay, start) - (y[0] - y[1])/elph_tau - y[0]*y[1]/pol_tau),
            ((y[0] - y[1])/elph_tau - y[0]*y[1]/pol_tau),
            (y[0]*y[1]/pol_tau)]

    return dydt

def twoStateModel(params, show_params, wav, data=None, plot=False):
    """Two state model used for fitting transient absorption data.
    Can plot measured data to compare w/ solution.
    Can also check how fit is going with show_params."""

    parvals = params.valuesdict()
    dt = parvals['dt']
    elph_tau = parvals['elph_tau']
    pol_tau = parvals['pol_tau']
    amp_ct = parvals['amp_ct']
    amp_p = parvals['amp_p']
    delay = parvals['delay']
    start = parvals['start']
    end = parvals['end']
    step = parvals['step']
    y0 = parvals['y0']
    y1 = parvals['y1']
    y2 = parvals['y2']

    if show_params is True:

        print ('%f, %f, %f, %f, %f, %f' %(elph_tau, pol_tau, amp_ct, amp_p, delay, dt))

    tspan = np.linspace(start, end, step)
    yinit = [y0, y1, y2]
    sol = solve_ivp(lambda t,y:rate(t, y, dt, elph_tau, pol_tau, delay, start),
                    [tspan[0], tspan[-1]], yinit, t_eval=tspan)

    if plot is True:

        return modelPlot(tspan, data, amp_ct, amp_p, sol, wav)

    return sol

def twoStateModel2(elph_tau, pol_tau, amp_ct, amp_p, delay, dt, start, end,
                   step, coeff, times, wav):
    """Used to guess fits by hand."""

    tspan = np.linspace(start, end, step)
    yinit = [0, 0, 0]
    sol = solve_ivp(lambda t,y:rate(t, y, dt, elph_tau, pol_tau, delay, start),
                    [tspan[0], tspan[-1]], yinit, t_eval=tspan)

    return modelPlot(times, coeff, amp_ct, amp_p, sol, wav)

def modelPlot(times, coeff, amp_ct, amp_p, sol, wav):
    """Plots the amplitude coefficients and the model fits."""

    xlabel = 'Time (fs)'
    ylabel = 'Coeff. Amplitude (Arb. Units)'
    xlim = [times[0], times[-1]]
    ylim = [None, 1.5]
    plt.figure('Two State %s' %wav)
    ax = plt.axes()
    ax.plot(sol.t, amp_ct*sol.y[0], '-', c='lightskyblue', lw=3, label='Charge Transfer State')
    ax.plot(sol.t, amp_ct*sol.y[1], 'k--', dashes=(7, 5))
    ax.plot(sol.t, amp_p*sol.y[2], '-', c='navajowhite', lw=3, label='Polaron State')
    ax.plot(times, coeff[0], 'o', c='steelblue')
    ax.plot(times, coeff[1], 's', c='darkorange')
    ax.legend(loc='best')
    cp.simple(ax, xlabel, ylabel, xlim, ylim)

def residualTwoTemp(params, t, data, show_params, wav):
    """Residual of two state model minus the data. Minized by the result()
    function."""

    amp_ct = params['amp_ct'].value
    amp_p = params['amp_p'].value
    sol = twoStateModel(params, show_params, wav)
    model = np.array([amp_ct*sol.y[0], amp_p*sol.y[2]])

    return (model - data).ravel()

def paramsTwoState(elph_tau, pol_tau, amp_ct, amp_p, delay, dt, start, end, step):
    """Sets up the parameters used in the two state model. elph_tau is limited by
    instrument time resolution so it's fixed. Amplitudes for the two states
    are needed to relate the transient absorption to the populations."""

    params = lmfit.Parameters()
    params.add('elph_tau', value = elph_tau, vary = False) #el-ph scattering time const.
    params.add('pol_tau', value = pol_tau) #polaron formation time const.
    params.add('amp_ct', value = amp_ct) #amplitude for CT state
    params.add('amp_p', value = amp_p) #amplitude for polaron state
    params.add('y0', value = 0, vary = False) #initial value for el pop.
    params.add('y1', value = 0, vary = False) #initial value for ph pop.
    params.add('y2', value = 0, vary = False) #initial value for pol pop.
    params.add('delay', value = delay) #pulse delay from 0
    params.add('dt', value = dt, vary = False) #pulse width
    params.add('start', value = start, vary = False) #start time for fit
    params.add('end', value = end, vary = False) #end time for fit
    params.add('step', value = step, vary = False) #step size for times in fit

    return params

def resultTwoState(params, tspan, interp_data, show_params, wav):
    """Does the actual least squares fit using lmfit package, where results is
    a lmfit minimizer object. Uses the fit parameters in the two state model."""

    mini = lmfit.Minimizer(residualTwoTemp, params, fcn_args=(tspan, interp_data,
                                                              show_params, wav))
    results = mini.minimize(method='leastsq', maxfev=100)
    data_fitted = twoStateModel(results.params, show_params=False, wav=wav)
    results.params.pretty_print()

    return results, data_fitted

def fitTwoState(elph_tau, pol_tau, amp_ct, amp_p, delay, dt, start, end, step,
                TA_file, init_time, fin_time, wav, n=1, semilog=False,
                show_params=False):
    """Overall fit function. Sets up parameters, fits two state model to data
    provided. Data should be a numpy array, it is then interpolated to 1000
    points to improve fit. Can show fit params while it runs.
    Plots fit results when done."""

    measured, times = decomposition(TA_file, init_time, fin_time, wav, semilog, n)
    tspan = np.linspace(start, end, step)
    interp_ct = np.interp(tspan, times, measured[0])
    interp_p = np.interp(tspan, times, measured[1])
    interp_data = np.array([interp_ct, interp_p])
    params = paramsTwoState(elph_tau, pol_tau, amp_ct, amp_p, delay, dt, start,
                            end, step)
    results, data_fitted = resultTwoState(params, tspan, interp_data, show_params, wav)
    r_params = results.params
    amp_ct = r_params['amp_ct'].value
    amp_p = r_params['amp_p'].value
    amp_ct_err = r_params['amp_ct'].stderr
    amp_p_err = r_params['amp_p'].stderr
    pol_tau = r_params['pol_tau'].value
    pol_tau_err = r_params['pol_tau'].stderr
    pol_prob = amp_p/amp_ct
    covar_ctp = results.covar[1][2]
    pol_prob_err = pol_prob*np.sqrt((amp_p_err/amp_p)**2
                                    +(amp_ct_err/amp_ct)**2
                                    -2*(covar_ctp/(amp_p*amp_ct)))
    #pol_prob_err = amp_p_err/amp_ct_err
    modelPlot(times, measured, amp_ct, amp_p, data_fitted, wav)
    print('Polaron formation prob.: %f +/- %f \n' %(pol_prob, pol_prob_err))
    return (pol_tau, pol_tau_err), (pol_prob, pol_prob_err)

def paramsMarcus(A, E_rel, T):
    """Sets up parameters for Marcus theory rate fit. To simplify equation,
    E_rel in this function is actually (E_g + E_rel)."""

    params = lmfit.Parameters()
    params.add('A', value = A, min = 80, max = 110) #Parameter related to el-ph coupling
    params.add('E_rel', value = E_rel, min = 2.0, max = 3.0) #E_g is 2.1 eV for a-Fe2O3
    params.add('T', value = T, min = 300, max = 600) #Temperature in K

    return params

def residualMarcus(params, energies, data=None, weights=None):
    """Calculates the residuals for Marcus theory fit. Can also be used to
    just return the function without fitting. Weights are errors."""

    parvals = params.valuesdict()
    A = parvals['A']
    T = parvals['T']
    E_rel = parvals['E_rel']
    kb = 8.61e-5
    func = (A*np.exp(((energies-E_rel)**2)/(4*E_rel*kb*T))) #Fitting to time constant so 1/rate
    if data is None:
        return func
    if weights is None:
        return (func - data)
    if weights is not None:
        weights = [(2*weight)**2 for weight in weights]
        return (func - data)/weights

def resultMarcus(params, energies, data, weights = None, interp = None):
    """Fits the polaron formation rates from two state model """

    if interp is not None:
        espan = np.linspace(2.21,2.58,10)
        interp_data = np.interp(espan, energies, data)
        plt.scatter(espan,interp_data)
        results = lmfit.minimize(residualMarcus, params, method='leastsq',
                                 maxfev=50, nan_policy = 'propagate',
                                 args = (espan, interp_data))

    results = lmfit.minimize(residualMarcus, params, method='leastsq', maxfev=50,
                             nan_policy = 'propagate', args = (energies, data, weights))
    return results

def fitMarcus(A, E_rel, T, energies, data, weights=None, linestyle=None):
    """Overall fit function for Marcus model"""

    espan = np.linspace(2.0, 3.5)
    params = paramsMarcus(A, E_rel, T)
    results = resultMarcus(params, energies, data, weights)
    data_fit = residualMarcus(results.params, espan)

    plt.figure('Marcus')
    ax = plt.axes()
    xlabel = 'Energy (eV)'
    ylabel = 'Formation Time (fs)'
    xlim = [espan[0], espan[-1]]
    ylim = [round(min(data)-20.1, -1), round(max(data)+10.1, -1)]
    cp.simple(ax, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    plt.plot(espan, data_fit, color='gray', lw=2, linestyle=linestyle)
    plt.errorbar(energies, data, weights, fmt='s', color='steelblue', capsize=3)
    plt.show()

def stretchedAmplitude(TA_file, pol_time, peak_min, peak_max, time_zero=None):
    """Takes in TA data and picks out the peak for the polaron state and returns
    the average value so it can be used for fitting function. Must make sure to
    pick the right energy range, peak_min and max are in eV."""

    TA = np.load(TA_file)
    times = TA[1:,0]
    if time_zero is not None:
        times -= time_zero
    times /= 1000
    energies = TA[0,1:]
    delA = TA[1:,1:]
    pol_state = delA[pol_time]
    idxmin = (np.abs(energies-peak_min)).argmin()
    idxmax = (np.abs(energies-peak_max)).argmin()
    peak_mean = np.array([np.mean(delA[i,idxmax:idxmin]) for i, time in enumerate(times)])
    peak_max = peak_mean.argmax()
    peak_mean = peak_mean[peak_max:]/peak_mean[peak_max]
    times = times[peak_max:]
    plt.figure('Peak Find')
    plt.clf()
    ax = plt.axes()
    plt.semilogx(times, peak_mean, 's')
    ax.set_xlabel('Times (ps)')
    ax.set_ylabel('Norm. $\Delta$A (Arb. Units)')
    plt.figure('Polaron')
    ax = plt.axes()
    plt.plot(energies, pol_state*1000)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('$\Delta$Abs. (mOD)')
    return times, peak_mean

def stretchedExp(params, times):
    """Stretched exponential function"""

    parvals = params.valuesdict()
    tau = parvals['tau']
    beta = parvals['beta']
    A = parvals['A']
    func = A*np.exp(-(times/tau)**beta)

    return func

def residualStretch(params, data, times):
    """Residual for stretched exponential fit"""

    stretched = stretchedExp(params, times)
    return (stretched - data)

def fitStretchedExp(A, tau, beta, data, times):
    """Takes in long time scale data and fits to stretched exponential.
    Times must be converted to ps first."""

    params = lmfit.Parameters()
    params.add('tau', value = tau, min=0) #Lifetime in ps
    params.add('beta', value = beta, min=0, max=1) #Stretching constant
    params.add('A', value = A, min=0, max=2) #Amplitude
    results = lmfit.minimize(residualStretch, params, args=(data, times))
    data_fit = stretchedExp(results.params, times)
    r_params = results.params
    lifetime = r_params['tau'].value
    lifetime_err = r_params['tau'].stderr
    hopping_r = r_params['beta'].value
    hopping_r_err = r_params['beta'].stderr
    if hopping_r_err == None:
        hopping_r_err = 0
    if lifetime_err == None:
        lifetime_err = 0
    print(results.params.pretty_print(),'\n')

    return (hopping_r, hopping_r_err), (lifetime, lifetime_err), data_fit

def contourTheory(GS, ES, shift_re, shift_th, shift_pol1, shift_pol2, times=None):
    """Takes in calculated spectra for ground and excited states and plots
    the transient absorption expected for different scenarios post excitation."""

    energies = np.linspace(45, 65, len(GS)) #energy in eV
    CT_TA = 0.8*ES - GS
    trip_exp = lambda t: np.array([(np.exp(-t/100) + np.exp(-t/1000)
                                    + np.exp(-t/10000))/3])
    #Excitation no shift
    ct_only = [CT_TA*trip_exp(time) for time in times[6:]]
    #Excitation with renormalization
    ct_renorm = [(ES*trip_exp(time) + (np.roll(.8*GS, shift_re))*(1-trip_exp(time)))
             - GS for time in times[6:]]
    #Thermal Expansion
    thermal_ta = [((np.roll(GS, 2*shift_th))
                    - 1.2*GS)*trip_exp(time) for time in times[6:]]
    #Trap state
    gaussian = lambda e_res, energy, dG: np.exp(-np.log(2)*((2*(e_res - energy)/dG)**2))
    trap = gaussian(53.5, energies, 0.5)
    trap_ta = [CT_TA*(np.exp(-time/100)+np.exp(-time/1000))/2 - (0.3*trap)
                *(1-(np.exp(-time/100)+np.exp(-time/1000))/2) for time in times[6:]]
    #Polaron formation
    polaron = np.roll(GS, shift_pol1) + np.roll(GS, shift_pol2)/3 + 2*GS/3
    polaron_norm = polaron/max(polaron)
    pol_ta = [CT_TA*np.exp(-time/100) + (0.9*polaron_norm - GS)
              *(1-trip_exp(time)) for time in times[6:]]

    excited_states = {'CT Only':ct_only, 'CT Norm.':ct_renorm, 'GS Thermal':thermal_ta,
                      'Trap State':trap_ta, 'CT to Polaron':pol_ta}
    for state_id, state in excited_states.items():
        ta_times = np.zeros([6, len(energies)])
        ta_times = np.append(ta_times, state, axis=0)

        plt.figure('%s' %state_id)
        ax = plt.axes()
        ax.contourf(energies, times, ta_times, levels=100, vmin=-.4, vmax=.4,
                     cmap='RdYlBu')
        cp.simple(ax, 'Energy (eV)', 'Time (fs)', xlim=[51, 59])

def sticksBroadening(sticks_file, energies, sig_G, dLmin, dLcutoff, dLscale, q,
                     comment, norm=True, plot=True, fan=False, gauss=False,
                     lor=False, save=False):
    """Takes in stick.xy file from CTM4XAS program and broadens them. Analytical
    formula for Fano convolved with Gaussian taken from S. Schippers paper."""

    with open(sticks_file) as f:
        sticks = []
        stick = False
        for line in f:
            if 'R' in line: #normal sticks file has redundant sticks
                break
            if 'Sticks' in line: #excited state file only has one row of sticks
                stick = True
                continue
            if stick is True:
                if not line.strip():
                    continue
                else:
                    sticks.append(list(map(float, line.split())))
        sticks = np.asarray(sticks)

    broad = np.zeros(len(energies))
    sln2 = np.sqrt(np.log(2))
    dG = 2*sig_G*np.sqrt(2*np.log(2))
    dL = lambda energy: max(dLmin, dLmin+(energy-dLcutoff)/dLscale)
    gaussian = lambda e_res, energy, dG: np.exp(-np.log(2)*((2*(e_res - energy)/dG)**2))
    lorentzian = lambda e_res, energy, dL: (1 + ((e_res - energy)/dL)**2)**-1
    fano = lambda e_res, q, energy, dL: ((1/(1+q**2))
                                        *((q*dL/2 + energy - e_res)**2)
                                        /((dL/2)**2 + (energy - e_res)**2))
    for e_res in sticks:
        x = 2*sln2*(e_res[0] - energies)/sig_G #Paper claims it's dG but only works if it's sigma.
        y = 2*dL(e_res[0])*sln2/dG
        z = x + 1j*y
        w = sc.special.wofz(z)
        if lor is True:
            broad += e_res[1]*lorentzian(e_res[0], energies, dLmin)
        elif gauss is True:
            broad += e_res[1]*gaussian(e_res[0], energies, dG)
        elif fan is True:
            broad += e_res[1]*fano(e_res[0], q, energies, dL(e_res[0]))
        else:
            #Default broadening. Analytical formula for a Fano convolved with a Gaussian
            broad += (e_res[1]
                      *((2*sln2)/(dG*np.sqrt(np.pi)))
                      *((1-1/q**2)*w.real - (2/q)*w.imag))
    if norm is True:
        broad -= min(broad)
        broad /= max(broad)
    if plot is True:
        plt.figure('Broad')
        plt.plot(sticks[:,0], sticks[:,1])
        plt.plot(energies, broad)
    if save is True:
        np.savetxt('%s_broad.txt' %comment, [energies, broad])
    return broad

short_waves = [['560nm.npy', [40,100,2,1,30,11,38,'560']],
               ['520nm2.npy', [40,100,2,1,30,11,38,'520']],
               ['480nm.npy', [40,100,2,1,30,11,38,'480']],
               ['400nm.npy', [40,100,2,1,30,11,38,'400']]]
long_waves = [['stretched_560nm.npy', [30, 48, 49, 2994, 1, 200, 1, '560']],
              ['stretched_520nm.npy', [30, 48, 49, 2994, 1, 200, 1, '520']],
              ['stretched_480nm.npy', [30, 48, 49, 1019, 1, 200, 1, '480']],
              ['stretched_400nm2.npy', [30, 48, 49, -206331, 1, 200, 1, '400']]]
def fitAll(short_waves=None, long_waves=None, amp=None, e_res=None, T=None,
           full_file=False):
    """Takes in TA files for all wavelengths and performs all analysis used in
    polaron paper."""
    os.chdir('C:\\Users\\LucasC\\Documents\\mystuff\\Data_allwave\\All waves')
    uv_vis = np.load('UV-vis.npy')
    if short_waves:
        pol_times = []
        pol_probs = []
        energies = []
        for short_file, guess in short_waves:
            elph_tau = guess[0]
            pol_tau = guess[1]
            amp_ct = guess[2]
            amp_p = guess[3]
            delay = guess[4]
            init_time = guess[5]
            fin_time = guess[6]
            wav = guess[7]
            print(wav+'nm')
            pol_time, pol_prob = fitTwoState(elph_tau, pol_tau, amp_ct, amp_p,
                                              delay, 50, -200, 3000, 1000, short_file,
                                              init_time, fin_time, wav, n=1,
                                              semilog=False, show_params=False)
            pol_times.append(pol_time)
            pol_probs.append(pol_prob)
            energies.append(1240/int(wav))

        pol_probs = np.asarray(pol_probs)
        pol_times = np.asarray(pol_times)
        plt.figure('Formation Probability')
        ax = plt.axes()
        ax.plot(uv_vis[:,0], uv_vis[:,1], color='dimgray', lw=2, zorder=0)
        ax.bar(energies, pol_probs[:,0], width=0.15, yerr=pol_probs[:,1],
               capsize=3, edgecolor='k', color='lightpink', ecolor='gray')
        ax.set_xlim(2.0, 3.5)
        plt.axvline(x=2.1, color='grey', linestyle='dashed', lw=.75)
        plt.axvline(x=2.8, color='grey', linestyle='dashed', lw=.75)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Formation probability')
        fitMarcus(amp, e_res, T, energies[:3], pol_times[:3,0], pol_times[:3,1])
        plt.axvline(x=2.1, color='grey', linestyle='dashed', lw=.75)
        plt.axvline(x=2.8, color='grey', linestyle='dashed', lw=.75)
        fitMarcus(amp, e_res, T, energies, pol_times[:,0], pol_times[:,1],
                  linestyle='dashed')

    if long_waves:
        hopping_rs = []
        lifetimes = []
        energies = []
        i = 0
        colors = {'560':['darkorange','navajowhite'], '520':['crimson','lightcoral'],
                  '480':['steelblue','lightskyblue'], '400':['k','lightgrey']}
        plt.figure('StretchExp')
        ax = plt.axes()
        for long_file, guess in long_waves:
            pol_time = guess[0]
            peak_min = guess[1]
            peak_max = guess[2]
            time_zero = guess[3]
            if full_file is True:
                times, data = stretchedAmplitude(long_file, pol_time, peak_min,
                                                 peak_max, time_zero)
            else:
                data_load = np.load(long_file)
                times = data_load[:,0]
                data = data_load[:,1]

            A = guess[4]
            tau = guess[5]
            beta = guess[6]
            wav = guess[7]
            print(wav+'nm')
            hopping_r, lifetime, data_fit = fitStretchedExp(A, tau, beta, data,
                                                            times)
            hopping_rs.append(hopping_r)
            lifetimes.append(lifetime)
            energies.append(1240/int(wav))
            plt.semilogx(times, 0.2*i + data, 's', color=colors[wav][1])
            plt.semilogx(times, 0.2*i + data_fit, color=colors[wav][0], lw=2)
            i += 1
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.set_xlabel('Times (ps)')
            ax.set_ylabel('Norm. $\Delta$A (Arb. Units)')

        hopping_rs = np.asarray(hopping_rs)
        lifetimes = np.asarray(lifetimes)
        plt.figure('Hopping Radius')
        ax = plt.axes()
        ax.plot(uv_vis[:,0], uv_vis[:,1], color='dimgray', lw=2, zorder=0)
        plt.errorbar(energies, hopping_rs[:,0], yerr=hopping_rs[:,1], fmt='s',
                     color='k', capsize=3)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Hopping radius, $\\beta$')
        plt.axvline(x=2.1, color='grey', linestyle='dashed', lw=.75)
        plt.axvline(x=2.8, color='grey', linestyle='dashed', lw=.75)
        ax.set_xlim(2.0, 3.5)
        ax.set_ylim(0.0, 1.2)

        plt.figure('Lifetimes')
        ax = plt.axes()
        ax.plot(uv_vis[:,0], 100 + (450)*uv_vis[:,1], color='dimgray', lw=2,
                zorder=0)
        ax.bar(energies, lifetimes[:,0], width=0.15, yerr=lifetimes[:,1],
               capsize=3, edgecolor='k', color='wheat', ecolor='gray')
        ax.set_xlim(2.0, 3.5)
        ax.set_ylim(100, 600)
        plt.axvline(x=2.1, color='grey', linestyle='dashed', lw=.75)
        plt.axvline(x=2.8, color='grey', linestyle='dashed', lw=.75)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Carrier lifetime (ps)')
