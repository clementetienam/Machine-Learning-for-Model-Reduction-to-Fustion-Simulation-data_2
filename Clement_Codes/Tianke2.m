function [aupdated,bupdated]=Tianke2(N,ytrain,simulated,alpha,aensemble,bensemble);

% N - size of ensemble
history=size(ytrain,2);

Sim11=reshape(simulated,5,history,N);

%History matching using ESMDA
for i=1:history % for the case when we have multi outputs
 fprintf('Now assimilating timestep %d .\n', i);
Sim1=Sim11(:,i,:);
Sim1=reshape(Sim1,5,N);

f=ytrain(:,i);
[aensemble,bensemble] = Tianke3 (aensemble,bensemble, f, N, Sim1,alpha);

 fprintf('Finished assimilating timestep %d \n', i);
end

aupdated=aensemble;
bupdated=bensemble;

 disp('  program executed  ');
end
 