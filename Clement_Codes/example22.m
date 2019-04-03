%% Applying Polynomial Kernel (Poly2)
%%

dd=importdata('chi_itg.dat',' ',1);
test=dd.data;
out=test;
outgp=test;
test=out;
X=log(test(1:600000,1:10));
y=(test(1:600000,11));
outputtest=y(290000+1:end,:);
y2=zeros(600000,1);
for i=1:600000
    if y(i)==0
        y2(i)=2;
    end
    
    if y(i)>0
        y2(i)=1;
    end
        
end
y=y2;
inputtrainclass=X(1:290000,:);
outputtrainclass=y(1:290000,:);
inputtest=X(290000+1:end,:);

Xtrain=inputtrainclass;
Ytrain=outputtrainclass;

layers = [ ...
  imageInputLayer([10 29000 1], 'Name', 'input')
          convolution2dLayer(5, 20, 'Name', 'conv_1')
          reluLayer('Name', 'relu_1')
          convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
          reluLayer('Name', 'relu_2')
          convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
          reluLayer('Name', 'relu_3')
          additionLayer(2,'Name', 'add')
          fullyConnectedLayer(10, 'Name', 'fc')
          softmaxLayer('Name', 'softmax')
          classificationLayer('Name', 'classoutput')];
options = trainingOptions('sgdm');
rng('default')
net = trainNetwork(Xtrain',Ytrain',layers,options);



EVALdaal = Evaluate(outputtrainclass,labelDA);
disp(['accuracy of the classifier is  = ' num2str(EVALdaal(:,1))]);