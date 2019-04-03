function [inputtrainGP,outputtrainGP,inputtrainclass,outputtrainclass,inputtest]=DatasetRegressClass()
%% GP classification/Regression model for the Chi data
disp(' We will implement the sparse GP together with classification for the Chi model' );
disp( 'Author: Dr Clement Etienam' )
disp( 'Supervisor: Professor Kody Law' )
disp( 'Supervised classfication Modelling' )
set(0,'defaultaxesfontsize',20); format long


dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
%X=zscore(test(1:600000,1:10));
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
outputtest=y(290000+1:end,:);
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=-1;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);

outputtrainclass=y(1:290000,:);
inputtest=X(290000+1:end,:);
%outputtest=y(290000+1:end,:);
p=10;
Matrixdata=[inputtrainclass outputtrainclass];

GPmatrix=[inputtrainclass (test(1:290000,11))];
outgp=GPmatrix;
outgp(any(outgp==0,2),:) = [];

outputtrainGP=log(outgp(:,11));
inputtrainGP=(outgp(:,1:10));

end

