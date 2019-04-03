function [y,x]=this

N=1000;

x=2*(rand(N,1)-0.5);
%x=4*pi*(rand(N,1)-0.5);

y=f(x);

figure
plot(x,y,'x');

figure
hist(y,100);

x;


function out = f(x)

out = (1+x).*(x<0) + x.*(x>=0);
%out = cos(x).*(x<0) -cos(x).*(x>=0);
