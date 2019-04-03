function rvec= cal_rl(rho,f)

    nrho = len(rho);
    rvec = zeros(nrho);

    for i=1:nrho-1
        rvec(i) = ( f(i+1) - f(i-1) ) / ( rho(i+1) - rho(i-1) ) / f(i);
    if  i== 0
        rvec(i) = ( f(i+1) - f(i) ) / ( rho(i+1) - rho(i) ) / f(i);
    end
    
    if i==-1 
        rvec(i) = ( f(i) - f(i-1) ) / ( rho(i) - rho(i-1) ) / f(i);
    end

    end