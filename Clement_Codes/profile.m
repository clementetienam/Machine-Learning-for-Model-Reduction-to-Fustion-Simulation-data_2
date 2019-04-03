function y= profile(nx, xmid, xwidth, yped, ysep, yaxis, alpha, beta, ytop)

    xped = xmid-xwidth/2;
    xtop = xmid-xwidth;
    
    a0 = (yped-ysep)/(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(xmid-0.5*xwidth-xmid)/xwidth));
    a1 = yaxis - ysep - a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(0.0-xmid)/xwidth));
    if ytop > 0.0
        yy = ysep + a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(xtop-xmid)/xwidth));
        a1 = (ytop - yy)/(1.0-(xtop/xped)^alpha)^beta;
    end
    
    x = linspace(1,nx)/(nx-1.0);

    y_edge = ysep + a0*(tanh(2.0*(1-xmid)/xwidth)-tanh(2.0*(x-xmid)/xwidth));
    
    y_core = zeros(nx);
    if yaxis > 0.0 ||ytop > 0
        for k=1:length(x)
            if xval(k) < xped(k)
                y_core1 = a1*(1.0-(xval(k)/xped(k))^alpha)^beta;
            else
                y_core1 = 0.0;
            end
            y_core(k)=y_core1;
        end
    end
    
    y = y_edge+y_core;

    end