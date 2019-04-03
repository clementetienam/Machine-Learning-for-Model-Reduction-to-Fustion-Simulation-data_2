clc;
clear;
close all;
r = normrnd(0,1,[10,10]);
r=reshape(r,1,100);
%!C:\Python32\python.exe rescaledd.py r
input1=py.rescaledd.rescc(r);