function [ J,grad] = Costfuntion(Theta, ...
                        input_layer,...
                        hidden_layer,...
                        num_label,...
                        X, y,lambda)
                
%COSTFUNTION Summary of this function goes here
%   Detailed explanation goes here
J=0;

Theta1 = reshape(Theta(1:(input_layer+1)*hidden_layer), ...
                hidden_layer,input_layer+1);
         
Theta2 = reshape(Theta((input_layer+1)*hidden_layer+1:end), ...
                num_label,hidden_layer+1);



m = size(X,1);

X = [ones(m,1),X];% 5000 by 401
a1 = X'; % 401 by 5000
z2 = Theta1 * a1; %25 by 401  *  401 by  5000
a2 = Sigmoid(z2);
a2 = [ones(1,m);a2];% 26 by 5000
z3 = Theta2 * a2; % 10 by 26 * 26 by 5000
a3 = Sigmoid(z3);
H_theta = a3;

ey = eye(num_label);
label_Y = ey(:,y);

J = (1/m)*(sum(sum(-label_Y.*log(H_theta)-(1-label_Y).*log(1-H_theta))));

% regularization 
square_Theta1 = sum(Theta1.^2);
reg_Theta1 = sum(square_Theta1) - square_Theta1(1,1);

square_Theta2 = sum(Theta2.^2);
reg_Theta2 = sum(square_Theta2) - square_Theta2(1,1);

regular = (lambda/(2*m))*(reg_Theta1+reg_Theta2);

J = J + regular;



%%%    back propogation 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

z2=[ones(1,m);z2];%26 by 5000
delta3 = a3-label_Y;% 10 by 5000
delta2=Theta2' * delta3 .* SigmoidGrad(z2);
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
 
 
 Theta1_grad = (1/m)*Delta1 + (lambda/m)*Theta1; 
 Theta1_grad(:,1) = D1(:,1) ;

 
 Theta2_grad = (1/m)*Delta2 + (lambda/m)*Theta2;
 Theta2_grad(:,1) = D2(:,1);

 grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

