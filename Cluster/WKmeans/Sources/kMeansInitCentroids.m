function centroids = kMeansInitCentroids(X, K)

centroids = zeros(K, size(X, 2));
m = size(X,1);
rands = randperm(m,K);
centroids = X(rands,:);

end

