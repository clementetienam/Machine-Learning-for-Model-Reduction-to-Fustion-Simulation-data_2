function ee=qmhd(a0, r0, b0, ip, kappa, delta)
  eps = a0/r0;
  ee=5.*a0^2/r0*b0/ip*(1.+kappa^2*(1.+2.*delta^2-1.2*delta^3))/2.*(1.17-0.65*eps)/(1.-eps^2)^2;
end