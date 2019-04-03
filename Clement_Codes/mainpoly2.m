function Thetaupdated=mainpoly2(N,ytrain,simulated,alpha,Thetanow);

% N - size of ensemble
history=size(ytrain,2);
history2=size(ytrain,1);
Sim11=reshape(simulated,history2,history,N);

%History matching using ESMDA
for i=1:history % for the case when we have multi outputs
 fprintf('Now assimilating timestep %d .\n', i);
Sim1=Sim11(:,i,:);
Sim1=reshape(Sim1,history2,N);

f=ytrain(:,i);
Thetanow = mainpoly3 (Thetanow, f, N, Sim1,alpha);
 
 fprintf('Finished assimilating timestep %d \n', i);
end

Thetaupdated=Thetanow;

 disp('  program executed  ');
end
 