#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:08:50 2019

@author: knhan930

Chebyshev Polynomial Approximation
"""
import numpy as np
import matplotlib.pyplot as plt

def chebyshev_node(degree, start_point, end_point):
    half_dist = (end_point - start_point) / 2
    center_point = (start_point + end_point) / 2
    vDeg = np.arange(degree + 1) + 1
    vCheby_point_pre = np.cos((2 * vDeg - 1) * np.pi / (2 * (degree + 1)))
    vCheby_point = half_dist * vCheby_point_pre + center_point
    
    return vCheby_point

def lagrange_polynomial(degree, vX, vY, start_point):
    vC = np.zeros(degree + 1)
    vNC = np.zeros(degree + 1)
    
    if degree == 2:
        vC[2] =   vY[1]/(vX[1]-vX[2])/(vX[1]-vX[0]) \
                + vY[2]/(vX[2]-vX[1])/(vX[2]-vX[0]) \
                + vY[0]/(vX[0]-vX[1])/(vX[0]-vX[2])
        vC[1] = - vY[1]/(vX[1]-vX[2])/(vX[1]-vX[0])*vX[0] \
                - vY[1]/(vX[1]-vX[2])/(vX[1]-vX[0])*vX[2] \
                - vY[2]/(vX[2]-vX[1])/(vX[2]-vX[0])*vX[0] \
                - vY[2]/(vX[2]-vX[1])/(vX[2]-vX[0])*vX[1] \
                - vY[0]/(vX[0]-vX[1])/(vX[0]-vX[2])*vX[2] \
                - vY[0]/(vX[0]-vX[1])/(vX[0]-vX[2])*vX[1]
        vC[0] =   vY[1]/(vX[1]-vX[2])/(vX[1]-vX[0])*vX[2]*vX[0] \
                + vY[2]/(vX[2]-vX[1])/(vX[2]-vX[0])*vX[1]*vX[0] \
                + vY[0]/(vX[0]-vX[1])/(vX[0]-vX[2])*vX[1]*vX[2]
                
        vNC[0] = vC[2]
        vNC[1] = 2 * vC[2] * start_point + vC[1]
        vNC[2] = vC[2] * (start_point ** 2) + vC[1] * start_point + vC[0]
    
    elif degree == 1:
        vC[1] = (vY[1]-vY[0])/(vX[1]-vX[0])
        vC[0] = (vX[1]*vY[0]-vX[0]*vY[1])/(vX[1]-vX[0])
        
        vNC[0] = vC[1]
        vNC[1] = vC[1] * start_point + vC[0]
        
    return vNC

def polynomial_approximation(degree, start, end, func):
    vX = chebyshev_node(degree, start, end)
    vY = func(vX)
    vC = lagrange_polynomial(degree, vX, vY, start)
    
    number_of_points = 1000
    
    vP = np.arange(start, end, (end-start)/number_of_points)
    vY_ref = func(vP)
    vY_approx = np.polyval(vC, vP)
    vDiff = -np.log2(np.abs(vY_ref - vY_approx))
    plt.plot(vP, vDiff)
    max_error = np.min(vDiff)
    return (vC, max_error)

#-----------
# Example with sin(pi*x)
#-----------    
degree = 2
start_point = 0
end_point = 0.01
f = lambda x: np.sin(np.pi * x)

(coeff, error) = polynomial_approximation(2, 0, 0.01, f)

print('Degree of polynomial = {}'.format(degree))
print('Start point = {}'.format(start_point))
print('End Point = {}'.format(end_point))
print('Coeff = {}'.format(coeff))
print('maximum error in log2 scale = {}'.format(error))


