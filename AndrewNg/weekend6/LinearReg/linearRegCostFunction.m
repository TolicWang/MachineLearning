function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h_theta = X*theta;
squar_error = (1/(2*m))*(sum((h_theta-y).^2));
regular_term = (lambda/(2*m))*(sum(theta.^2) - theta(1)^2);
J = squar_error + regular_term;


grad(1) = (1/m)*sum((h_theta - y).*X(:,1));
for j = 2:size(theta,1)
    grad(j) =  (1/m)*sum((h_theta - y).*X(:,j))+(lambda/m)*theta(j);
end






% =========================================================================

grad = grad(:);

end
