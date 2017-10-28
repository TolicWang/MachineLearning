load('ex5data1.mat');
p = 3;
lambda  = 1;
% Training set
X_poly = polyFeatures(X,p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly,1), 1), X_poly];                   % Add Ones

% Validation set
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);% Normalize
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val]; 

m = size(X_poly,1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i = 1:m
    theta = trainLinearReg(X_poly(1:i,:),y(1:i),lambda);
    error_train(i) = linearRegCostFunction(X_poly(1:i,:),y(1:i),theta,0);
    error_val(i) = linearRegCostFunction(X_poly_val,yval,theta,0);
end

plot(1:m,error_train,1:m,error_val);
title(sprintf('Learning curve for M with Reg p = %f,lambda = %f',p,lambda))
legend('Train', 'Cross Validation')
xlabel('Number of training example')
ylabel('Error')
axis([0 13 0 20])



