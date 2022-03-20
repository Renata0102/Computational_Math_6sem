import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import scipy.optimize
from scipy.optimize import newton_krylov
import math

CAUCHY = [1.76 * 10**(-3), 0, 0, 0]


def func(y, param):
    A = 7.89 * 10**(-10)
    B = 1.1 * 10**(7)
    C = 1.13 * 10**(3)
    M = 10**(6)
    return([-A*y[0]-B*y[0]*y[2], A*y[0]-M*C*y[1]*y[2], A*y[0]-B*y[0]*y[2]-M*C*y[1]*y[2]+C*y[3], B*y[0]*y[2]-C*y[3]])


def RK2(start, stop, step, Cauchy, a, func):
    time = [0]
    solution = [np.array(Cauchy)]
    
    while time[-1] < stop:
        k1 = scipy.optimize.root((lambda k: func(solution[-1] + step * (2 + np.sqrt(2))/2 * k, a) - k), np.array(func(solution[-1],a))).x
        k2 = scipy.optimize.root((lambda k: func(solution[-1] + step * (-np.sqrt(2) * k1 + (2 + np.sqrt(2)/2) * k),a) - k),np.array(func(solution[-1],a))).x
        time.append(time[-1] + step)
        solution.append(solution[-1] + step/2 * (k1 + k2))
        
    return(time, solution)

def RK3(start, stop, step, Cauchy, a, func):
    time = [0]
    solution = [np.array(Cauchy)]
    
    while time[-1] < stop:
        k1 = scipy.optimize.root((lambda k: func(solution[-1] + step * (3 + np.sqrt(3))/6 * k, a) - k), func(solution[-1],a)).x
        k2 = scipy.optimize.root((lambda k: func(solution[-1] + step * ((3 + 2*np.sqrt(3))/3 * k1 + (3 + np.sqrt(3)/6) * k), a) - k),
                           func(solution[-1],a)).x
        time.append(time[-1] + step)
        solution.append(solution[-1] + step/2 * (k1 + k2))
    return(time, solution)

def FDN2(start, stop, step, Cauchy, a,func):
    time, solution = RK3(start, start + step, step, Cauchy, a, func)
    while time[-1] < stop:
        time.append(time[-1] + step)
        new_y = scipy.optimize.root(lambda y: step * np.array(func(y,a)) - 3/2 * y + 2* solution[-1] - 1/2 * solution[-2], solution[-1]).x
        solution.append(new_y)
    return(time, solution)

def FDN3(start, stop, step, Cauchy, a,func):
    time, solution = RK3(start, start + 2 * step, step, Cauchy, a, func)
    while time[-1] < stop:
        time.append(time[-1] + step)
        new_y = scipy.optimize.root(lambda y: (step * np.array(func(y,a))) - 11/6 * y + 3 * solution[-1] - 3/2 * solution[-2] + 1/3 * solution[-3], solution[-1]).x
        solution.append(new_y)
    return(time, solution)

def FDN4(start, stop, step, Cauchy, a, func):
    time, solution = RK3(start, start + 3 * step, step, Cauchy, a, func)
    while time[-1] < stop:
        time.append(time[-1] + step)
        new_y = scipy.optimize.root(lambda y: (step * np.array(func(y,a))) - 25/12 * y + 4 * solution[-1] - 3 * solution[-2] + 4/3 * solution[-3] - 1/4 * solution[-4], solution[-1]).x
        solution.append(new_y)
    return(time, solution)


def make_plots(method, stop, step):
    time, solution = method(0, stop, step, CAUCHY, True, func)
    solution = np.array(solution).T

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for i in range(2):
        for j in range(2):
            axs[i][j].plot(time, solution[2*i + j])
            axs[i][j].set_xlabel('$t$')
            axs[i][j].set_ylabel(f'$y_{2*i + j + 1}$')
            
            
def make_plots_nevyazka(methods_dict, stop, step):
    time, solution_FDN4 = methods_dict['FDN4'](0, stop, step, CAUCHY, True, func)
    for key, val in methods_dict.items():
        if key == 'FDN4':
            continue
        time, solution = val(0, stop, step, CAUCHY, True, func)
        t, n = [], [] 
        for i in range(stop):
            t.append(i * step)
            n.append(abs(solution[i][0] - solution_FDN4[i][0]))
        
        plt.plot(t, n, label = key)
        plt.xlabel("t")
        plt.ylabel("|r|")
        plt.legend()
    
    plt.show()