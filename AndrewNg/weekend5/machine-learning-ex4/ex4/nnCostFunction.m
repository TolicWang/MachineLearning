function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


   X = [ones(m,1),X]; % 5000 by 401
   a1 = X'; % 401 by 5000;  theta1 is 25 by 401
   z2 = Theta1*a1;% 25 by 5000
   a2 = sigmoid(z2); % 25 by 5000
   a2 = [ones(1,m);a2]; % 26 by 5000
   z3 = Theta2*a2;% theta2 is 10 by 26 ;  z3 is 10 by 5000
   a3 = sigmoid(z3);% 10 by 5000
   H_theta = a3;
      ey = eye(num_labels);
    label_Y = ey(:,y);
  
  J = (1/m)*(sum(sum(-label_Y.*log(H_theta)-(1-label_Y).*log(1-H_theta))));
    % 
    s_Theta1 = sum(Theta1 .^ 2);
  r_Theta1 = sum(s_Theta1)-s_Theta1(1,1);
  
  s_Theta2 = sum(Theta2 .^ 2);
  r_Theta2 = sum(s_Theta2)-s_Theta2(1,1);
  
    regular = (lambda/(2*m))*(r_Theta1+r_Theta2);
  
  J = J + regular;
      
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

z2=[ones(1,m);z2];%26 by 5000
delta3 = a3-label_Y;% 10 by 5000
delta2=Theta2' * delta3 .* sigmoidGradient(z2);
%         26 by 10    10 by 5000          26 by 5000 

delta2=delta2(2:size(delta2,1),:); % 26 by 5000 ==> 25 by 5000

%
Delta2 = zeros(size(Theta2)); % 10 by 26
Delta1=zeros(size(Theta1)); % 25 by 401
% 
% 
Delta2 =Delta2+ delta3*(a2)'; %  10by26  +  10 by 5000 *  5000 by 26
Delta1 =Delta1 + delta2*(a1)';% 25 by 401 + 25 by 5000 * 5000 by 401
% 
 D1 = (1/m)*Delta1;
 D2 = (1/m)*Delta2;
% 
Theta1_grad= D1;
Theta2_grad = D2;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
 Theta1_grad = (1/m)*Delta1 + (lambda/m)*Theta1;  % 25 by 401
 Theta1_grad(:,1) = D1(:,1) ;
 
 
  Theta2_grad = (1/m)*Delta2 + (lambda/m)*Theta2;% 10 by 26
 Theta2_grad(:,1) = D2(:,1);
 
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
