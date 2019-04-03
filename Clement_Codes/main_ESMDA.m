function [aupdated,bupdated]=main_ESMDA(N,ytrain,simulated,alpha,aensemble,bensemble);

% N - size of ensemble
history=size(ytrain,2);

Sim11=reshape(simulated,6,history,N);

%History matching using ESMDA
for i=1:history % for the case when we have multi outputs
 fprintf('Now assimilating timestep %d .\n', i);
Sim1=Sim11(:,i,:);
Sim1=reshape(Sim1,6,N);

f=ytrain(:,i);
[aensemble,bensemble] = ESMDA (aensemble,bensemble, f, N, Sim1,alpha);

 fprintf('Finished assimilating timestep %d \n', i);
end

aupdated=aensemble;
bupdated=bensemble;

 disp('  program executed  ');
end
 