%----------------------------------------------------------------------------------
% Surrogate fowarding of reservoir simulation
% Using Gaussian process
% Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
%-----------------------------------------------------------------------------------
%% 

clc
clear
% N - size of ensemble

N=input( ' enter the number of realizations(ensemble size)  '); %100
nx=input( ' enter the number of grid blocks in x direction  '); %84
ny=input( ' enter the number of grid blocks in y direction  '); %27
nz=input( ' enter the number of grid blocks in z direction  '); % 4
history=input(' enter the number of timestep for the history period '); %16

disp( 'Load the true permeability and porosity')
load rossmary.GRDECL; %true permeability field
sizetrue=numel(rossmary);
load rossmaryporo.GRDECL; %True porosity field
oldfolder=cd;
cd(oldfolder) % setting original directory
disp('  import the true observation data  ');
 
  True= importdata('Real.RSM',' ',7);
 
 True=True.data;
Time=True(:,1);
 
  TO1=True(:,6);
 TW1=True(:,8);
 TP1=True(:,9);

 
 
 disp(' make the true observation')
 for ihistory=1:history
 obs=zeros(3,1);
 obs(1,:)=TO1(ihistory,:);
 obs(2,:)=TW1(ihistory,:);
 obs(3,:)=TP1(ihistory,:);
 observation(:,ihistory)=obs;
 end
 %%
 disp('Learn the model using supervised learning to check how good')
 truefield=zeros(sizetrue,2);
 truefield(:,1)=log(rossmary);
 truefield(:,2)=rossmaryporo;
 [u,s,v]=svd(truefield,'econ');
 truefield=u(1:history,:)*s*v';
 %truefield=truefield(1:history,:);
 trueoutput=zeros(history,3);
 observ=observation';
 trueoutput(1:16,:)=observ;
 
 %rstool(truefield,trueoutput(:,1),'linear');
 for i=1:3
 mdl = fitlm(truefield,trueoutput(:,i),'linear','RobustOpts','on');
 ypred = predict(mdl,truefield);
 youtlv(:,i)=abs(ypred);
 end
 
  for i=1:3
 mdl = fitrgp(truefield,trueoutput(:,i),'KernelFunction','ardsquaredexponential',...
      'FitMethod','sr','PredictMethod','fic','Standardize',1);
 ypred = predict(mdl,truefield);
 youtrgp(:,i)=abs(ypred);
 end

%  for j=1:3
% xu = normrnd(mean(observ(:,j)),std(observ(:,j)),9056,1); 
% xsparse(:,j)=abs(xu);
%  end
 %trueoutput(17:end,:)=xsparse;
 for i=1:3
 ymuvv(:,i)= surrogate(truefield,trueoutput(:,i),truefield);
 end
 for i=1:3
 ymuvvspatial(:,i)= surrogate2(truefield,trueoutput(:,i),truefield);
 end
  linecolor1 = colordg(4);
 disp( 'compare eclipse run with GP surrogtae');
 %% Plot for Well Bottom Hole Pressure
figure()
 plot(Time,youtlv(1:history,1),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('Q_o(STB/DAY)','FontName','Helvetica', 'Fontsize', 13);
  %ylim([5000 25000])
hold on
plot(Time,observation(1,:),'r','LineWidth',1)

 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [5000 25000],'Color','black','LineStyle','--')
legend('Eclipse','surrogate','location','northeast');
hold off
%saveas(gcf,sprintf('PRO-Oilrate_iter%d.fig',iyobo))
close(figure)

figure()
 plot(Time,youtlv(1:history,2),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('Q_w(STB/DAY)','FontName','Helvetica', 'Fontsize', 13);
%ylim([1 200])
title('Producer Water production Rate','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,observation(2,:),'r','LineWidth',1)
set(gca,'yscale','log', 'FontName','Helvetica', 'Fontsize', 13)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [1 200],'Color','black','LineStyle','--')
legend('Eclipse','surrogate','location','northeast');
hold off
%saveas(gcf,sprintf('PRO-WATER_iter%d.fig',iyobo))
close(figure)

figure()
 plot(Time,youtlv(1:history,3),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('BHP(Psia)','FontName','Helvetica', 'Fontsize', 13);
 %ylim([1500 4000])
title('Injector BHP','FontName','Helvetica', 'Fontsize', 13)
hold on
plot(Time,observation(3,:),'r','LineWidth',1)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [1500 4000],'Color','black','LineStyle','--')

legend('Eclipse','surrogate','location','northeast');
hold off
%saveas(gcf,sprintf('Injector-WBHP_iter%d.fig',iyobo))
close(figure)
%%
% 
% oldfolder=cd;

% %% Creating Folders
% disp( 'create the folders')
% for j=1:N
% f = 'MASTER';
% folder = strcat(f, sprintf('%.5d',j));
% mkdir(folder);
% end
% 
% %% Coppying simulation files
% disp( 'copy simulation files for the forward problem')
% for j=1:N
% f = 'MASTER';
% folder = strcat(f, sprintf('%.5d',j));
% copyfile('FAULT.DAT',folder)
% copyfile('MASTER0.DATA',folder)
% copyfile('surrogate.m',folder)
% end
% %% Machine Learning part
% % disp( 'Loading the overcomplete dictionary of permeability')
% % load Yes2.out; %Permeability dictionary
% % load Yes2poro.out; %porosity dictionary
% 
% 
% %% The big history matching iterative loop will start here
% tic;
% %for iclement=1:alpha 
% %% Loading Porosity and Permeability ensemble files
% disp(' load the permeability and porosity fields')
% 
%     disp( 'permeability loaded from initial ensemble')
% load sgsimporo.out; %initial porosity
% load sgsim.out; %initial permeabiity
% 
% 
% cd(oldfolder) % setting original directory
% 
% %% Saving POROVANCOUVER and KVANCOUVER
% 
% for i=1:N %list of folders 
%     
%     f = 'MASTER';
%    folder = strcat(f, sprintf('%.5d',i));
%    
%     cd(folder) % changing directory 
%     
%     PORO2=poro(:,i);
%     PERMX2=perm(:,i);
%    
%     
%     save('KVANCOUVER.DAT','PERMX2','-ascii');
%     save('POROVANCOUVER.DAT','PORO2','-ascii');
%     
%     cd(oldfolder) % returning to original cd
%     
% end
% 
% %% Inserting KEYWORDS PORO and PERMY 
% 
% parfor i=1:N %list of folders 
%     
%     f = 'MASTER';
%     folder = strcat(f, sprintf('%.5d',i));
%     cd(folder)
% 
% CStr = regexp(fileread('KVANCOUVER.DAT'), char(10), 'split');
% CStr2 = strrep(CStr, 'toReplace', 'Replacement');
% CStr2 = cat(2, {'PERMY'}, CStr2(1:end));
% CStr2 = cat(2, CStr2(1:end), {'/'});
% FID = fopen('KVANCOUVER.DAT', 'w');
% if FID < 0, error('Cannot open file'); end
% fprintf(FID, '%s\n', CStr2{:});
% fclose(FID);
% 
% CStr = regexp(fileread('POROVANCOUVER.DAT'), char(10), 'split');
% CStr2 = strrep(CStr, 'toReplace', 'Replacement');
% CStr2 = cat(2, {'PORO'}, CStr2(1:end));
% CStr2 = cat(2, CStr2(1:end), {'/'});
% FID = fopen('POROVANCOUVER.DAT', 'w');
% if FID < 0, error('Cannot open file'); end
% fprintf(FID, '%s\n', CStr2{:});
% fclose(FID);
% 
% cd(oldfolder) % setting original directory
% 
% end
% 
% 
% %% Running Simulations
% disp( 'Solve the Non-Linear fluid flow Forward Problem' )
% cd(oldfolder) % setting original directory
% 
% parfor i=1:N %list of folders 
%     
%     f = 'MASTER';
%     folder = strcat(f, sprintf('%.5d',i));
%     cd(folder)   
% 
%  ymuv= surrogate(truefield,trueoutput,realisation)
%  
%  ymuvall(:,:,i)=ymuv;
% 
%     cd(oldfolder) % setting original directory
%     
% end
% disp(' plot production profile for the run')
% index=plotproduction(N,iclement);
% %end