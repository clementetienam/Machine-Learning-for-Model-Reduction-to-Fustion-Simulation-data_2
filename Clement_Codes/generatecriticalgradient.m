clc;
clear;
close all;
%%
mu0 = 4.0*pi*1.0e-7;
a0 = r0/A;
ip = ip_q95(q95, a0, r0, b0, kappa0, delta0);
ngw = ip/(pi*a0^2)*10.0;
neped = ngw * fgw_ped;
neaxis = nepeak*neped;
c_betan = 4.0*1.602e5*neped*mu0/b0^2*(a0*b0/ip);
teped = betan_ped/c_betan;
teaxis = betan_axis/c_betan;
rho = linspace(0,1.,nrho);
rhob = sqrt(kappa0)*a0*rho;
a = a0*rho;
kappa = 0.5*(1.0+kappa0+(kappa0-1.0)*rho^2);
delta = delta0*rho^2;
vol = pi*2.0*pi*a^2*kappa*(r0-0.25*a*kappa);
nrho= 101;
r0 = 1.65;
A= 3.0;
%%
        b0         = 1.75;
        kappa0     = 1.8;
        delta0     = 0.6;
        q95        = 5.0;
        xwid       = 0.075;
        fgw_ped    = 0.5;
        nepeak     = 1.7;
        ne_alpha   = 1.5;
        ne_beta    = 1.5;
        betan      = 3.0;
        betan_axis = 5.0;
        betan_ped  = 0.7;
        te_alpha   = 1.5;
        te_beta    = 1.5;
        tau        = 0.8;
        zeff0      = 1.5;
        qmin       = 1.0;

        %%
nx = nrho;
xmid = 1.0-0.5*xwid;
xwidth = xwid;
yped = neped;
ysep = 0.25*neped;
yaxis = neaxis;
alpha = ne_alpha;
beta = ne_beta;
ne = profile(nx,xmid , xwidth , yped, ysep,yaxis,alpha,  beta);
yaxis = teaxis;
alpha = te_alpha;
beta = te_beta;
te = profile(nx,1.0-0.5*xwid, xwidth , yped, 0.075,yaxis,alpha,  beta);

    ti = te/tau;

    ni = ne;
    nz = zeros(nrho);

    zeff = zeff0*ones(nrho);
    nbeam = zeros(nrho);

    q = qmin+(q95-qmin)*rho^2;
    shat = cal_rl(rho,q)*rho + 1.0e-6;
    
%%    
    a0 = r0/A;

    ip = ip_q95(q95, a0, r0, b0, kappa0, delta0);

    ngw = ip/(pi*a0^2)*10.0;
    neped = ngw * fgw_ped;
    neaxis = nepeak*neped;

    c_betan = 4.0*1.602e5*neped*mu0/b0^2*(a0*b0/ip);
    teped = betan_ped/c_betan;
    teaxis = betan_axis/c_betan;
    %%
    
    rho = linspace(0,1.,nrho);
    rhob = sqrt(kappa0)*a0*rho;
    a = a0*rho;
    kappa = 0.5*(1.0+kappa0+(kappa0-1.0)*rho^2);
    delta = delta0*rho^2;
    vol = pi*2.0*pi*a^2*kappa*(r0-0.25*a*kappa);
    
    %%
    nx = nrho;
    xmid = 1.0-0.5*xwid;
    xwidth = xwid;
    yped = neped;
    ysep = 0.25*neped;
    yaxis = neaxis;
    alpha = ne_alpha;
    beta = ne_beta;
    ne = profile(nx,xmid , xwidth , yped, ysep,yaxis,alpha,  beta);
    yaxis = teaxis;
alpha = te_alpha;
beta = te_beta;
te = profile(nx,1.0-0.5*xwid, xwidth , yped, 0.075,yaxis,alpha,  beta);
  ti = te/tau;

    ni = ne;
    nz = zeros(nrho);

    zeff = zeff0*ones(nrho);
    nbeam = zeros(nrho);

    q = qmin+(q95-qmin)*rho^2;
    shat = cal_rl(rho,q)*rho + 1.0e-6;
   %%
    w = 1.5*1.602e3*(ne*te+(ni+nz)*ti);
     for i=1:numel(w)-1
    wsuma = (vol(i+1)-vol(i))*w(i);
        wsumaa(i)=wsuma;
     end 
        wsum=sum(wsumaa);
    betan_cal = wsum/vol(end)/1.5;
    betan_cal = 2.0*mu0/b0^2 *betan_cal;
    betan_cal = betan_cal/fabs(ip/(a0*b0));
    betan_cal = 1.0e2*betan_cal;
    
  
    %%
        yaxis = teaxis;
alpha = te_alpha;
beta = te_beta;
te = profile(nx,1.0-0.5*xwid, xwidth , yped, 0.075,yaxis,alpha,  beta);

ti = te/tau;

    rlni = r0*abs(cal_rl(rho,ne));
    rlti = r0*abs(cal_rl(rho,ti));

%% call cratical gradient model

    chi = forwarding(ne, te, ti, zeff, q, shat, rlni, rlti, a, r0, 1., 1.0);