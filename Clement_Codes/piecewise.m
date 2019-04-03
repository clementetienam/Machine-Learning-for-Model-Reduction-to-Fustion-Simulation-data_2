function y=piecewise(x)
y=zeros(size(x,1),1);
for i=1:numel(x)
if (x(i)>=-10 || -5>x(i))
    y(i)=sin(x(i));
elseif (x(i)>=-5 || 10>x(i))
        y(i)=exp(-0.5*(x(i)).^2);
elseif (x(i)>=10 || 12>=x(i))
        y(i)=1;
elseif x(i)>12
    y(i)=-1;
else
    y(i)=-2;
end
end
end
            
  