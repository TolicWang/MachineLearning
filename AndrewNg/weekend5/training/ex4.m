%%===================training neural network============= 
%
%   there is a neural network with 3 layers - an input layer, a hidden
%   layer and an output layer. The training exaple is a matrix 5000 by 400.
%
%   S1 = 400;S2 = 25; S3 = 10
%   Theta1 has size 25 by 401
%   Theta2 has size 10 by 26

input_layer = 400;
hidden_layer = 25;
num_label = 10;

load('ex4data1.mat');
%load('ex4weights.mat');
m = size(X,1);

Theta1 = InitialTheta(hidden_layer,input_layer+1);%  25 by 401
Theta2 = InitialTheta(num_label,hidden_layer+1);% 10 by 26
%     gradient checking 
lambda = 1;
Theta = [Theta1(:);Theta2(:)];

%GradientChecking(Theta,input_layer,hidden_layer,num_label,X,y,lambda)
%  if gradient checking is correce, turn off it




alpha = 1.5;

iteration = 50; 
%===================training by gradient descent
J_history = zeros(iteration,1);
fprintf('Training by gradient descent......\n');
for i=1:iteration
    %% cos J will mostly reach the optimum, when i si equal to 2500
    % and the  Accuracy will be about 98%
    [J,grad] = Costfuntion(Theta,input_layer,hidden_layer,...
                                    num_label,X,y,lambda);
    J_history(i)=J;
    Theta = GradientDescent(Theta,grad,alpha,m,lambda);
   
    Theta1 = reshape(Theta(1:hidden_layer*(input_layer+1)), ...
        hidden_layer,(input_layer+1));
    Theta2 = reshape(Theta((1+hidden_layer*(input_layer+1)):end), ...
        num_label,(hidden_layer+1));
    
end
pred1 = predict(Theta1, Theta2, X);
mean1 = mean(double(pred1 == y)) * 100;
%===================training by gradient descent


%===============training by fmincg====================
fprintf('Training by fmincg......\n');
costFunc = @(Theta)Costfuntion(Theta, input_layer, hidden_layer, ...
                               num_label, X, y, lambda);
options = optimset('MaxIter',iteration);
[Theta, cost] = fmincg(costFunc, Theta, options);

 Theta1 = reshape(Theta(1:hidden_layer*(input_layer+1)), ...
        hidden_layer,(input_layer+1));
 Theta2 = reshape(Theta((1+hidden_layer*(input_layer+1)):end), ...
        num_label,(hidden_layer+1));
    
pred2 = predict(Theta1, Theta2, X);
mean2 = mean(double(pred2 == y)) * 100;
%=============================================


figure(1);
hold on;
plot(J_history,'r','MarkerSize',20);
plot(cost,'b','MarkerSize',20);
xlabel('iteration');
ylabel('J');
legend('Gradient descent','fmincg');



fprintf('\nTraining Set Accuracy of Gradient descent: %f\n', mean1);
fprintf('\nTraining Set Accuracy of fmingc: %f\n', mean2);

