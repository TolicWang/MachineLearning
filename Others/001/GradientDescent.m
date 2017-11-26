function GradientDescent(x1,x2)
grad = zeros(2,1);

x= [x1;x2];
alpha = 0.00025;

for i = 1:500
    grad(1)=4*x1-2*x2-4;
    grad(2)=-2*x1+10*x2-24;
    x = x-alpha*grad;
    f1 = f(x(1),x(2));
end
fprintf('GradientDescent:\n');
fprintf('x1=%f\tx2=%f\tv=%f\n',x(1),x(2),f1);

end


