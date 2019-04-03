#!/usr/bin/env python
"""
 -----------------------------------------------------------------------
 Generate training data for chi with critical graident model 
 based on a modified IFS-PPPL ITG model
 -----------------------------------------------------------------------
"""

import sys
from numpy import *
from random import uniform
from critical_gradient import chi_itg

mu0 = 4.0*pi*1.0e-7

def qmhd(a0, r0, b0, ip, kappa, delta):

    eps = a0/r0
    return 5.*a0**2/r0*b0/ip*(1.+kappa**2*(1.+2.*delta**2-1.2*delta**3))/2.*(1.17-0.65*eps)/(1.-eps**2)**2

def ip_q95(q95, a0, r0, b0, kappa, delta):

    eps = a0/r0
    return 5.*a0**2/r0*b0/q95*(1.+kappa**2*(1.+2.*delta**2-1.2*delta**3))/2.*(1.17-0.65*eps)/(1.-eps**2)**2

def profile(nx, xmid, xwidth, yped, ysep, yaxis, alpha, beta, ytop=0):

    xped = xmid-xwidth/2
    xtop = xmid-xwidth
    
    a0 = (yped-ysep)/(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(xmid-0.5*xwidth-xmid)/xwidth))
    a1 = yaxis - ysep - a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(0.0-xmid)/xwidth))
    if ytop > 0.0:
        yy = ysep + a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(xtop-xmid)/xwidth))
        a1 = (ytop - yy)/(1.0-(xtop/xped)**alpha)**beta
    
    x = arange(nx)/(nx-1.0)
    y_edge = ysep + a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(x-xmid)/xwidth))
    
    y_core = zeros(nx)
    if yaxis > 0.0 or ytop > 0:
        for k,xval in enumerate(x):
            if xval < xped: y_core[k] = a1*(1.0-(xval/xped)**alpha)**beta
            else: y_core[k] = 0.0
    
    y = y_edge+y_core

    return y

def cal_rl(rho,f):

    nrho = len(rho)
    rvec = zeros(nrho)

    for i in range(1,nrho-1):
        rvec[i] = ( f[i+1] - f[i-1] ) / ( rho[i+1] - rho[i-1] ) / f[i]
    i= 0; rvec[i] = ( f[i+1] - f[i] ) / ( rho[i+1] - rho[i] ) / f[i]
    i=-1; rvec[i] = ( f[i] - f[i-1] ) / ( rho[i] - rho[i-1] ) / f[i]

    return rvec

def model_chi (
        nrho       = 101,
        r0         = 1.65, 
        A          = 3.0,
        b0         = 1.75, 
        kappa0     = 1.8,
        delta0     = 0.6,
        q95        = 5.0,
        xwid       = 0.075,
        fgw_ped    = 0.5,
        nepeak     = 1.7,
        ne_alpha   = 1.5,
        ne_beta    = 1.5,
        betan      = 3.0,
        betan_axis = 5.0,
        betan_ped  = 0.7,
        te_alpha   = 1.5,
        te_beta    = 1.5,
        tau        = 0.8,
        zeff0      = 1.5,
        qmin       = 1.0,
        file       = None):

    #--- post input

    a0 = r0/A

    ip = ip_q95(q95, a0, r0, b0, kappa0, delta0)

    ngw = ip/(pi*a0**2)*10.0
    neped = ngw * fgw_ped
    neaxis = nepeak*neped

    c_betan = 4.0*1.602e5*neped*mu0/b0**2*(a0*b0/ip)
    teped = betan_ped/c_betan
    teaxis = betan_axis/c_betan

    # print 'q95 = ', q95
    # print 'ip = ', ip
    # print 'ngw = ', ngw
    # print 'neped = ', neped
    # print 'neaxis = ', neaxis
    # print 'teped = ', teped
    # print 'teaxis = ',teaxis

    #--- analytic metric

    rho = linspace(0,1.,nrho)
    rhob = sqrt(kappa0)*a0*rho
    a = a0*rho
    kappa = 0.5*(1.0+kappa0+(kappa0-1.0)*rho**2);
    delta = delta0*rho**2
    vol = pi*2.0*pi*a**2*kappa*(r0-0.25*a*kappa)

    #--- model density, temperature profiles

    ne = profile(
             nx = nrho, 
             xmid = 1.0-0.5*xwid, 
             xwidth = xwid, 
             yped = neped, 
             ysep = 0.25*neped,
             yaxis = neaxis, 
             alpha = ne_alpha, 
             beta = ne_beta)

    te = profile(
             nx = nrho, 
             xmid = 1.0-0.5*xwid, 
             xwidth = xwid, 
             yped = teped, 
             ysep = 0.075,
             yaxis = teaxis, 
             alpha = te_alpha, 
             beta = te_beta)

    ti = te/tau

    ni = ne
    nz = zeros(nrho)

    zeff = zeff0*ones(nrho)
    nbeam = zeros(nrho)

    q = qmin+(q95-qmin)*rho**2
    shat = cal_rl(rho,q)*rho + 1.0e-6

    #--- scale temperature with given betan

    w = 1.5*1.602e3*(ne*te+(ni+nz)*ti)
    wsum = sum([(vol[i+1]-vol[i])*w[i] for i in range(len(w)-1)])
    betan_cal = wsum/vol[-1]/1.5
    betan_cal *= 2.0*mu0/b0**2
    betan_cal /= fabs(ip/(a0*b0))
    betan_cal *= 1.0e2

    #print 'betan = ', betan_cal
    #print 'scale = ', betan/betan_cal

    te = profile(
             nx = nrho, 
             xmid = 1.0-0.5*xwid, 
             xwidth = xwid, 
             yped = teped, 
             ysep = 0.075,
             yaxis = teaxis*betan/betan_cal, 
             alpha = te_alpha, 
             beta = te_beta)

    ti = te/tau

    rlni = r0*abs(cal_rl(rho,ne))
    rlti = r0*abs(cal_rl(rho,ti))

    #--- call cratical gradient model

    chi = chi_itg(ne, te, ti, zeff, q, shat, rlni, rlti, a, r0, Stiffness = 1., Alpha = 1.0)

    #--- write

    if file:

       for i in range( int(0.2*nrho), int(0.8*nrho) ):
           f.write("%12.3e"%ne[i])
           f.write("%12.3e"%te[i])
           f.write("%12.3e"%ti[i])
           f.write("%12.3e"%zeff[i])
           f.write("%12.3e"%q[i])
           f.write("%12.3e"%shat[i])
           f.write("%12.3e"%rlni[i])
           f.write("%12.3e"%rlti[i])
           f.write("%12.3e"%a[i])
           f.write("%12.3e"%r0)
           f.write("%12.3e"%chi[i])
           f.write("\n")

if __name__ == "__main__":

    ndata = int(sys.argv[1])

    f = open("chi_itg.dat","w")
    for var in ["ne","te","ti","zeff","q","shat","rlni","rlti","a","r0","chi"]:
        f.write("%12s"%var)
    f.write("\n")

    for k in range(ndata):

        print 'profile %10d'%k

        chi = model_chi(
            nrho       = 101,
            r0         = uniform(4.0, 8.0),
            A          = uniform(2.5, 3.5),
            b0         = uniform(4.0, 8.0),
            kappa0     = uniform(1.5, 2.0),
            delta0     = uniform(0.3, 0.6),
            q95        = uniform(3.0, 7.0),
            xwid       = 0.075,
            fgw_ped    = uniform(0.2, 0.3),
            nepeak     = uniform(1.2, 2.0), 
            ne_alpha   = uniform(1.0, 2.0),
            ne_beta    = uniform(1.0, 2.0),
            betan_axis = 3.0,
            betan_ped  = uniform(0.5, 1.0),
            betan      = uniform(2.0, 5.0),
            te_alpha   = uniform(1.0, 2.0),
            te_beta    = uniform(1.0, 2.0),
            tau        = uniform(0.8, 1.2),
            zeff0      = uniform(1.5, 2.5),
            qmin       = uniform(1.0, 2.5),
            file       = f
        )

