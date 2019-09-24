# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 07:24:22 2019

@author: jarl

Basically, one should obtain the unit circle from 0<Re<1/A; -0.5<Im<0.5 for the f(t) = exp(-t) ideally..
https://www.wolframalpha.com/input/?i=nyquist+plot+of+Laplace+transform+%28exp%28-t%29%29

"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import mpmath as mp
from utils import simpson_nonuniform_AitkensExtrap, dLaplace_dt
import time


# decleare floating point percition and the length of our time series (as discrete points)
mp.dps = 20
series_Length = 1000

# decleare function variables t = time; s = 1/t (frequency)
t, s = sym.symbols('t s')

# Any function may be used to test the model, however some topologies may cause numerical problems if too many points is
# placed on a symptotically converging region. This is due to the ZeroDivisions errors in Aitkens iterations ( currently
# this is handeled by applying A= +/- infinity)
F = sym.exp(t) 

# apply the selected complex numbers to test on the transfer function i.e. omega*i
omega = np.logspace(-2, 4, num=50)

# Since F is given symbolically, then the exact result can be derived by differential calculus
L_F = sym.laplace_transform(F, t, s)[0]
L_F_approx = sym.lambdify(s, L_F)
analytical_solution = L_F_approx(np.array([ complex(0, omega[i]) for i in range(len(omega))] ))


# Here, the numerical approximate begins, by taking creating a time series, of t and f(t) respectively
T =  np.array( mp.linspace(0, 30, series_Length) )
F_approx =  sym.lambdify(t, F, 'mpmath')




F_t = [0]* len(T)
for i in range(len(T)):
    F_t[i] = F_approx(T[i])
F_t = np.array( F_t )



time_calc = time.time()

SolutionVector  = [complex(0,0)]*len(omega)
for i in range(len(omega)):
    SolutionVector[i] = simpson_nonuniform_AitkensExtrap(T, [ dLaplace_dt(F_t[j], T[j], complex(0, omega[i]) ) for j in range(len(T))  ], maxAitkensIterations=int(series_Length/4 -1 ) )
    SolutionVector[i]  = complex( float(SolutionVector[i].real), float(SolutionVector[i].imag ))

print("Calculation time: " + str(time.time() - time_calc) + " seconds")

SolutionVector = np.array(SolutionVector)

SolutionVector_std = np.std(SolutionVector)
SolutionVector_mean = np.mean(SolutionVector)

SolutionVector_Filtered = []
for entry in SolutionVector:
    if abs(entry - SolutionVector_mean) < SolutionVector_std*2: # 95 % stat. significance
        SolutionVector_Filtered.append(entry)
SolutionVector_Filtered = np.array(SolutionVector_Filtered)



"""
Plotting results
1st: the time dependent function f(t)
2nd: The analytical and numerical approximation of in the Real-Complex plane
3rd: The corresponding approximation error of in the Real-Complex plane
"""



plt.plot(T, F_t, '-k')
plt.grid(b=True)
plt.ylabel('f(t)')
plt.xlabel('time (s)')
plt.legend('f(t) = ' + str(F) )
plt.show()


midpoint = int( len(SolutionVector)/2)

plt.plot( SolutionVector.real, SolutionVector.imag, '-b')
plt.plot( analytical_solution.real, analytical_solution.imag, '--r')
plt.legend([\
            'Simps + Aitkens', \
            'Analytical' \
            ])
plt.xlabel('\Re(L[f(t)](\omega i))')
plt.ylabel('\Im(L[f(t)](\omega i))')

plt.title('Numerical and Analytical solutions to' + 'L(' + str(F) + ') =' + str(L_F) )
plt.grid(b=True)
plt.show()


lap_Ait2_err = SolutionVector - analytical_solution

plt.plot( lap_Ait2_err.real, lap_Ait2_err.imag, '-m')


plt.legend([\
            'RMSE=' + str(abs( np.sqrt(np.mean(lap_Ait2_err**2)) )), \
            ]  )
plt.xlabel('\Re(L[f(t)](\omega i))')
plt.ylabel('\Im(L[f(t)](\omega i))')

plt.title('Numerical deviation to Simps + Aitken methods')
plt.grid(b=True)
plt.show()
