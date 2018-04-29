

load('s.mat')
nmi = 0;
save s.mat nmi


times  = 100;
for belta =2:5
    
for t = 1:times;
    fprintf('iteration %d.\n',t);
    fprintf('belta = %f\n',belta);
    s = run(belta);
    load('s.mat');
if nmi < s
    nmi = s;
    save s.mat nmi;
    save belta.mat belta;
end
end

end