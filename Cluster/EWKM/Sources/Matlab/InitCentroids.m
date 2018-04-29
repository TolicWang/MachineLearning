function centroids = InitCentroids(X, K)

centroids = zeros(K, size(X, 2));

m = size(X,1);

rands = randperm(m,K); 

centroids = X(rands,:);% randomly chose K point in X

end

