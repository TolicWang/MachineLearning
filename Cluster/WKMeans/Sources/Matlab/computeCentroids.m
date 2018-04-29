function centroids = computeCentroids(X, idx, K)
[m n] = size(X);

centroids = zeros(K, n);
Xy = [X idx];
for k = 1:K;
    
    index = find(Xy(:,n+1)==k);
    temp = X(index,:);
    s = sum(temp);
    centroids(k,:) = s./size(index,1);
end

end

