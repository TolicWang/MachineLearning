function diff = GradientChecking( Theta,input_layer, ...
                        hidden_layer,num_label, ...
                        X,y, ...
                        lambda)
                    
                    
 r = randperm(100,5);
 X = X(r,:);
 y = y(r,:);                 

costFunc = @(Theta)Costfuntion(Theta, input_layer, hidden_layer, ...
                               num_label, X, y, lambda);
                           
[cost,grad] = costFunc(Theta);

Theta1 = reshape(Theta(1:(input_layer+1)*hidden_layer), ...
                hidden_layer,(input_layer+1));
         
Theta2 = reshape(Theta((input_layer+1)*hidden_layer+1:end), ...
                num_label,hidden_layer+1);
            
 % compute the numarical gard
 %=====================================
 numarical_grad = zeros(size(Theta));
 temp = zeros(size(Theta));
 epsilon =1e-4;
 
 for i = 1:numel(Theta);% 
     temp(i) = epsilon;
     loss1 = costFunc(Theta - temp);
     loss2 = costFunc(Theta + temp);
     numarical_grad(i) = (loss2-loss1)/(2*epsilon);
     temp(i) = 0;
     
 end
 %=====================================

 % gard by back propogation
 %=====================================
 diff = norm(numarical_grad-grad)/norm(numarical_grad+grad);
 
 %=====================================

end

