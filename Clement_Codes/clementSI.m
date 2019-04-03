function loglikelihood=clementSI(Bb,Theta)
alpha=exp(Theta);
for i=1:size(Bb,1)
    r=Bb(i,:);
vrtheta=alpha(1,:).*((alpha(2,:)./r).^12-(alpha(2,:)./r).^6 );
vrtheta(isnan(vrtheta))=0;
vrthetasee(i,:)=vrtheta;
vrtheta1=sum(vrtheta);
vrthetabig(i,:)=vrtheta1;
end

marginalv=sum(vrthetabig)/size(Bb,1);
%disp('Check integral')
F = exp(-(vrthetasee));
for i=1:36
I(:,i) = trapz(F(:,i),vrthetasee(:,i));
end
victor=sum(I);
loglikelihood=marginalv+victor;
end
% I2 = trapz(F,2);
% victor2=sum(I2);
% loglikelihood2=marginalv+victor2;
% like = sum(log(eig(C(1:N,1:N)+sigg^2*eye(N)))) + ...
%     (y-X*beta)'*((C(1:N,1:N)+sigg^2*eye(N))\(y-X*beta));