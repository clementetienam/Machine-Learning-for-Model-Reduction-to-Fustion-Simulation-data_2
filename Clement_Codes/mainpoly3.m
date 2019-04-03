function Thetanow = mainpoly3 (Thetanow, f, N, Sim1,alpha)

%-----------------------------------------------------------------------------
disp('  generate Gaussian noise for the observed measurments  ');
    
   for i=1:size(f,1)
   stddWOPR1 = 0.1*f(i,:);
   stdall(i,:)=stddWOPR1;
   end


disp(' determine the standard deviation for measurement pertubation')
nobs = length(f);
noise = randn(max(10000,nobs),1);

disp('  generate Gaussian noise for the observed measurments  ');
Error1=stdall;


sig=Error1;
for i = 1 : length(f)
           f(i) = f(i) + sig(i)*noise(end-nobs+i);
end
R = sig.^2;
  Dj = repmat(f, 1, N);
           for i = 1:size(Dj,1)
             rndm(i,:) = randn(1,N); 
             rndm(i,:) = rndm(i,:) - mean(rndm(i,:)); 
             rndm(i,:) = rndm(i,:) / std(rndm(i,:));
             Dj(i,:) = Dj(i,:) + sqrt(alpha)*sqrt(R(i)) * rndm(i,:);
           end


Cd2 =diag(R);
disp('  generate the ensemble state matrix containing parameters and states  ');

overall=Thetanow;

Y=overall; 

M = mean(Sim1,2);

M2=mean(overall,2);

for j=1:N
    S(:,j)=Sim1(:,j)-M;
end
for j=1:N
    yprime(:,j)=overall(:,j)-M2;
end
disp('  update the new ensemble  ');
Cyd=(yprime*S')./((N-1));
Cdd=(S*S')./((N-1));
disp('  update the new ensemble  ');
Ynew=Y+(Cyd*pinv2((Cdd+(alpha.*Cd2))))*(Dj-Sim1);
%Ynew=Y+(Cyd/(Cdd+(alpha.*Cd)))*(Dj-Sim1);
disp( 'extract the active permeability field ')
Thetanow=Ynew;
end