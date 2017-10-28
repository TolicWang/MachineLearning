clear ; close all; clc
%% Step 0.    Loading data
% Load from ex5data1: 
load('ex5data1.mat');
% You will have X, y, Xval, yval, Xtest, ytest in your environment
%% Step 1.     Visualzing 
Visualizing(X,y);
pause;
fprintf('Program paused. Press enter to continue.\n');
%% Step 2.    Model selection
learningCurveP;
pause;
fprintf('Program paused. Press enter to continue.\n');
% learningCurveMwithoutReg;  %Analyzing
% pause;
learningCurveLambda;
pause;
fprintf('Program paused. Press enter to continue.\n');
% learningCurveMwithReg; %Analyzing
% pause;

p = 3;
lambda = 1;

%Training set
X_poly = polyFeatures(X,p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(size(X_poly,1), 1), X_poly];                   % Add Ones
% Test set
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);% Normalize
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];

%% Step 3. Estimate generization error
theta = trainLinearReg(X_poly,y,lambda);
J_error = linearRegCostFunction(X_poly_test,ytest,theta,0);
fprintf('J_error = %f\twhen p = %f\tlambda = %f\n',J_error,p,lambda);



