function [x]=thisforx

N=1000;
pkg load statistics
x=2*(rand(N,1)-0.5);
%x=4*pi*(rand(N,1)-0.5);

y=f(x);

figure
plot(x,y,'x');

figure
hist(y,100);

x;