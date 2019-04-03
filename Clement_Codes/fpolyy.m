function y2 = fpolyy(X,y,Theta)
y2 = sum(((X*Theta(:,1))+(X.^2*Theta(:,2))+(X.^3*Theta(:,3))-y));