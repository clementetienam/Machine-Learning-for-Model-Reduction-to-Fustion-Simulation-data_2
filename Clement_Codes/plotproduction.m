function index=plotproduction(N,iyobo)

oldfolder=cd;
cd(oldfolder) % setting original directory
 %% Plot the Production profile of ensemble
disp('  start the plotting  ');

    for i=1:N %list of folders 
    
    f = 'MASTER';
    folder = strcat(f, sprintf('%.5d',i));
    
    cd(folder);
    A1 = importdata('MASTER0.RSM',' ',7);
  
    
    A1=A1.data;
    
    
     WOPR1=A1(:,3);
    
     WWPR1=A1(:,5);
     WBHP1=A1(:,6);
     
     Time=A1(:,1);

    WOPRA(:,i)=WOPR1;
    WWPRB(:,i)=WWPR1;
    WBHPC(:,i)=WBHP1;
    
   
    
    cd(oldfolder);
    end
   cd(oldfolder) % returning to original directory
 %Import true data
 True= importdata('Real.RSM',' ',7);
 
 
 True=True.data;

 
 TO1=True(:,6);
 TO2=True(:,8);
 TO3=True(:,9);
 
 
 %grey = [0.4,0.4,0.4]; 
 linecolor1 = colordg(4);
 
 
%% Plot for Well Bottom Hole Pressure
figure()
 plot(Time,WOPRA(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('Q_o(STB/DAY)','FontName','Helvetica', 'Fontsize', 13);
  ylim([5000 25000])
title('Producer Oil production Rate','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');

hold on
plot(Time,TO1,'r','LineWidth',1)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [5000 25000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True model','Realisations','location','northeast');
hold off
saveas(gcf,sprintf('PRO-Oilrate_iter%d.fig',iyobo))
close(figure)

figure()
 plot(Time,WWPRB(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('Q_w(STB/DAY)','FontName','Helvetica', 'Fontsize', 13);
ylim([1 200])
title('Producer Water production Rate','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,TO2,'r','LineWidth',1)
b = get(gca,'Children');
set(gca,'yscale','log', 'FontName','Helvetica', 'Fontsize', 13)
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [1 200],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True model','Realisations','location','northeast');
hold off
saveas(gcf,sprintf('PRO-WATER_iter%d.fig',iyobo))
close(figure)

figure()
 plot(Time,WBHPC(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Time (days)','FontName','Helvetica', 'Fontsize', 13);
ylabel('BHP(Psia)','FontName','Helvetica', 'Fontsize', 13);
 ylim([1500 4000])
title('Injector BHP','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(Time,TO3,'r','LineWidth',1)
b = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
line([1500 1500], [1500 4000],'Color','black','LineStyle','--')
h = [b;a];
legend(h,'True model','Realisations','location','northeast');
hold off
saveas(gcf,sprintf('Injector-WBHP_iter%d.fig',iyobo))
close(figure)


for i=1:N
    EWOP1(i,:)=immse(WOPRA(:,i),TO1);
    EWOP2(i,:)=immse(WWPRB(:,i),TO2);
    EWOP3(i,:)=immse(WBHPC(:,i),TO3);
   
end
TOTALERROR=ones(N,1);
TOTALERROR=(EWOP1./std(TO1))+(EWOP2./std(TO2))+(EWOP3./std(TO3));
   
   
TOTALERROR=TOTALERROR./16;
jj=min(TOTALERROR);
index = TOTALERROR; 
bestnorm = find(index == min(index));
	%Pssim = Pnew(:,bestssim); %best due to ssim
fprintf('The best Norm Realization for production data match is number %i with value %4.6f \n',bestnorm,jj);
% JOYLINE=[1:100]';
% figure()
%bar(JOYLINE,TOTALERROR);

reali=[1:N]';

 figure()
 bar(reali,index,'cyan');
 xlabel('Realizations', 'FontName','Helvetica', 'Fontsize', 13);
 ylabel('RMSE value', 'FontName','Helvetica', 'Fontsize', 13);
 title('Production data Cost function for Realizations','FontName','Helvetica', 'Fontsize', 13)
 set(gcf,'color', 'white');
 hold on
 scatter(reali,index,'black','filled');
  xlim([1,N]);
 xlabel('Realizations', 'FontName','Helvetica', 'Fontsize', 13)
 ylabel('RMSE value', 'FontName','Helvetica', 'Fontsize', 13)
  saveas(gcf,sprintf('RMS_iter%d.fig',iyobo))
close(figure)

 disp('  program almost executed  ');


end