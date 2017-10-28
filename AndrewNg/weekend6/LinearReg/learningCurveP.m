
load('ex5data1.mat');
p = [1;2;3;4;5;6;7;8;9;10];
P = size(p,1);
error_train = zeros(P,1);
error_val = zeros(P,1);
for i = 1:P
    X_poly = polyFeatures(X,p(i));
    [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
    X_poly = [ones(size(X_poly,1), 1), X_poly];                   % Add Ones

    % Validation set
    X_poly_val = polyFeatures(Xval, p(i));
    X_poly_val = bsxfun(@minus, X_poly_val, mu);
    X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);% Normalize
    X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val]; 

    theta = trainLinearReg(X_poly,y,0);
    error_train(i) = linearRegCostFunction(X_poly,y,theta,0);
    error_val(i) = linearRegCostFunction(X_poly_val,yval,theta,0);

end

%% plot

plot(1:P,error_train,1:P,error_val);
title('Learning curve for P')
legend('Train', 'Cross Validation')
xlabel('Number of p')
ylabel('Error')
axis([0 13 0 30])
