function Theta=gradientadaptive2(a,b,Theta,iterations,alpha);
        X=reshape(a,1,10);
        Y=b;
 for i =1:iterations
        %Loss = Y - np.dot(X,Theta) + (np.dot(Theta.T,Theta)*0.001) ;
        Loss=Y-(X*Theta)+((Theta'*Theta)*0.001);
        Loss = Loss*(-1);
        %dJ = (np.dot(X.T,Loss)*2)/len(Y);
        dJ=((X'*Loss)*2)/length(Y);
   

        Theta = Theta - (alpha*dJ);
 end
  