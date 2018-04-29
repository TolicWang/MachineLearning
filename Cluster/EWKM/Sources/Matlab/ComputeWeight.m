function lambda= ComputeWeight( X,idx,gamma,centroids)
K = size(centroids,1);
[m,n] = size(X);
lambda = zeros(K,n);
D = zeros(K,n);
Xy = [X,idx];

for k =1:K
    %% computing D
    %  in each iteration, computing the kth row of D.
     index = find(Xy(:,n+1)==k)';% firstly,find all points's index belong to kth cluster
    temp = X(index,:);% take out all points belong to kth cluster from X
    square = (temp - centroids(k,:)) .^2;
    D(k,:) = sum(square);
end
  

for l = 1:K
    %% computing lambda
    % in each ieteration, computing the kth row of lambda.
    numerator = exp(-D(l,:) ./ gamma);
    denominator = sum(numerator);
    lambda(l,:) = numerator ./ denominator;
 
end
end
