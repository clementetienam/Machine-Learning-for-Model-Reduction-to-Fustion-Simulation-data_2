function chi=forwarding(ne, te, ti, zeff, q, shat, rlni, rlti, a, r0, Stiffness, Alpha )
   tau = ti/te;

    mu = 1.0/q;

    nu = 0.84*2.5*r0*ne/(ti*te^3)^0.5;

    rlti_cr = 2.46*(1.+2.78*mu^2)^0.26*(zeff/2.)^0.7*tau^0.52 ...
            *( (0.671+0.570*shat-0.189*rlni)^2 +0.335*rlni+0.392-0.779*shat+0.210*shat^2)...
            *( 1.-0.942*(2.95*(a/r0)^1.257/nu^0.235-0.2126)...
            *zeff^0.516 / abs (shat)^0.671);

    ddd=(rlti/rlti_cr-1.0);
            if ddd<0 
                dxc=0;
            end
            if ddd>0
                dxc=1;
            end
            if ddd==0
                dxc=0.5;
            end
    chi =  Stiffness*(rlti - rlti_cr)^Alpha *dxc;
end