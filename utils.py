# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 06:38:47 2019

@author: jarl

Utilities for numerical integration, and integral extrapolation to infinity (Aitken), e.g. int_0^oo

"""
import numpy as np
import mpmath as mp
import sympy as sym

def simpson_nonuniform(x, f):
    """
    copy pasta from: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    Simpson rule for irregularly spaced data.

        Parameters
        ----------
        x : list or np.array of floats
                Sampling points for the function values
        f : list or np.array of floats
                Function values at the sampling points

        Returns
        -------
        float : approximation for the integral
    """
    # For Quality assurance, lets add an assert:
    assert(len(x) == len(f) )
    N = len(x) - 1
    h = np.diff(x)

    result = 0.0
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += f[i] * ( h[i]**3 + h[i - 1]**3
                           + 3. * h[i] * h[i - 1] * hph )\
                     / ( 6 * h[i] * h[i - 1] )
        result += f[i - 1] * ( 2. * h[i - 1]**3 - h[i]**3
                              + 3. * h[i] * h[i - 1]**2)\
                     / ( 6 * h[i - 1] * hph)
        result += f[i + 1] * ( 2. * h[i]**3 - h[i - 1]**3
                              + 3. * h[i - 1] * h[i]**2)\
                     / ( 6 * h[i] * hph )

    if (N + 1) % 2 == 0:
        result += f[N] * ( 2 * h[N - 1]**2
                          + 3. * h[N - 2] * h[N - 1])\
                     / ( 6 * ( h[N - 2] + h[N - 1] ) )
        result += f[N - 1] * ( h[N - 1]**2
                           + 3*h[N - 1]* h[N - 2] )\
                     / ( 6 * h[N - 2] )
        result -= f[N - 2] * h[N - 1]**3\
                     / ( 6 * h[N - 2] * ( h[N - 2] + h[N - 1] ) )
    return result    

def AitkensExtrapolation(a, MaxIter):
    """
    A simple implementation of aitkens extrapolation (Aitken's delta-squared process)
        a : series of an integral sum, or something that is assumingly converging.
        
        TODO: prevent division by zero errors and other approximation errors
    """
    N = len(a)
    assert(type(MaxIter) == int)
    assert(N - 2*MaxIter > 0.0 )
    
    if len(a) < 3 -0.01 :
        return a[-1]
    elif MaxIter > 1.01 and N > 3 + 0.01:
        while N > 3 -0.01 and MaxIter > 1.01 :
            A = [0]*(N-2) 
            for i in range(N - 2):
                denom = (a[i]+a[i+2]-2*a[i+1])
                if denom == 0.0:
                    A[i] = mp.sign( a[i]*a[i+2] - a[i+1]**2 ) * mp.sign(denom) * mp.inf
                    print("Warning: Denominator of zero at iteration n=" + str( (len(a) - N)/2) + " i="+ str(i) + "  ; Setting A_n[i] = " + str(A[i])  )
                else:
                    A[i] = ( a[i]*a[i+2] - a[i+1]**2 ) /(a[i]+a[i+2]-2*a[i+1])
                
                    
            
            # prep for new cycle
            N -= 2
            a = A
            MaxIter -= 1
        return A[-1]
    elif MaxIter == 1 and N > 3 - 0.01:
        denom = (a[-3]+a[-1]-2*a[-2])
        if denom == 0.0:
            print("Warning: Denominator of zero at iteration N=" + 1 + " i="+ str(len(a)-2) + "  ; Setting A_N[i] = " + str(mp.inf)  )
            return mp.sign( a[-3]*a[-1] - a[-2]**2 ) * mp.sign(denom) * mp.inf
        else:
            return ( a[-3]*a[-1] - a[-2]**2 ) /(a[-3]+a[-1]-2*a[-2]) 
    else:
        return a[-1]

def simpson_nonuniform_AitkensExtrap(x, f, maxAitkensIterations = 0):
    """
    copy pasta from: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    Simpson rule for irregularly spaced data.

        Parameters
        ----------
        x : list or np.array of floats
                Sampling points for the function values
        f : list or np.array of floats
                Function values at the sampling points

        Returns
        -------
        float : approximation for the integral
    """
    # For Quality assurance, lets add an assert:
    assert(len(x) == len(f) )
    N = len(x) - 1
    h = np.diff(x)
    
    if (N + 1) % 2 == 0:
        cumulative_sum = [0]*(int(N/2)+1)
    else:
        cumulative_sum = [0]*int(N/2)
    
    result = 0.0
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += f[i] * ( h[i]**3 + h[i - 1]**3
                           + 3. * h[i] * h[i - 1] * hph )\
                     / ( 6 * h[i] * h[i - 1] )
        result += f[i - 1] * ( 2. * h[i - 1]**3 - h[i]**3
                              + 3. * h[i] * h[i - 1]**2)\
                     / ( 6 * h[i - 1] * hph)
        result += f[i + 1] * ( 2. * h[i]**3 - h[i - 1]**3
                              + 3. * h[i - 1] * h[i]**2)\
                     / ( 6 * h[i] * hph )
        
        cumulative_sum[int( (i-1)/2 ) ] = result
        

    if (N + 1) % 2 == 0:
        result += f[N] * ( 2 * h[N - 1]**2
                          + 3. * h[N - 2] * h[N - 1])\
                     / ( 6 * ( h[N - 2] + h[N - 1] ) )
        result += f[N - 1] * ( h[N - 1]**2
                           + 3*h[N - 1]* h[N - 2] )\
                     / ( 6 * h[N - 2] )
        result -= f[N - 2] * h[N - 1]**3\
                     / ( 6 * h[N - 2] * ( h[N - 2] + h[N - 1] ) )
        
        cumulative_sum[-1] = result
        
    return AitkensExtrapolation(cumulative_sum, maxAitkensIterations)

def Arg (complexNum):
    if complexNum.imag != 0:
        return 2*np.arctan( ( complexNum.real**2 + complexNum.imag**2 - complexNum.real) / complexNum.imag )
    elif complexNum.real == 0 and complexNum.imag == 0 :
        return np.nan
    elif complexNum.real > 0:
        return 0.0
    elif complexNum.real < 0:
        return np.pi
    



def pushFiniteArray (array, size_n, push_elem):
    if len(array) < size_n -0.01:
        return array.append(push_elem)
    else:
        return array[1:].append(push_elem)


# Time and frequency symbolic variables
t, s = sym.symbols('t s')


# The laplace integrand    
dLaplace_dt = lambda f_t, t, s : f_t * mp.exp(-s * t) 

# Redundant integration formulas based on midpoint and
# trapziodal rules, simpsons integration is supperior to these
"""
LaplaceTrans = lambda f_t, t, dt, s : dLaplace_dt(f_t, t, s)*dt

DiscreteLaplaceTrans_midPoint = lambda f_t_array, t_array, dt_array, s : sum( np.array( [ LaplaceTrans(f_t=f_t_array[i], t=t_array[i], dt=dt_array[i], s=s) for i in range(len(t_array)) ])  )

DiscreteLaplaceTrans_Trapz = lambda f_t_array, t_array, dt_array, s : 0.5 * ( LaplaceTrans(f_t=f_t_array[0], t=t_array[0], dt=dt_array[0], s=s) \
                                                                        + LaplaceTrans(f_t=f_t_array[-1], t=t_array[-1], dt=dt_array[-1], s=s) \
                                                                           + 2* sum( np.array( [ LaplaceTrans(f_t=f_t_array[i], t=t_array[i], dt=dt_array[i], s=s) for i in range(1, len(t_array)) -1 ])  ) )    

"""




