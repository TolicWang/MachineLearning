function CoordinateAscent(x1,x2)
f1 = f(x1,x2);
accur = 1.0e-5;
while(1)
    x1=1+0.5*x2;
    x2=2.4+0.2*x1;
    f2 = f(x1,x2);
    if abs(f1-f2)<accur
        break;
    end
  
    f1 = f2;
end
fprintf('CoordinateAscent:\n');
fprintf('x1=%f\tx2=%f\tv=%f\n',x1,x2,f2,i);
end