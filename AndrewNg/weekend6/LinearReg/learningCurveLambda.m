load('ex5data1.mat');
p = 3;
lambda_vec = [0;0.001;0.003;0.01;0.03;0.1;0.3;1;3;10];
% Training set
X_poly = polyFeatures(X,p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly,1), 1), X_poly];                   % Add Ones

% Validation set
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);% Normalize
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val]; 

len = size(lambda_vec,1);
error_train = zeros(len, 1);
error_val   = zeros(len, 1);
for i = 1:len
    lambda = lambda_vec(i);
    theta = trainLinearReg(X_poly,y,lambda);
    error_train(i) = linearRegCostFunction(X_poly,y,theta,0);
    error_val(i) = linearRegCostFunction(X_poly_val,yval,theta,0);
end
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
