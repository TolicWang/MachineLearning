function centroids = ComputeCentroids( X,idx,K)
[m,n] = size(X);
centroids = zeros(K,n);
Xy = [X,idx];

for k = 1:K
    %in each iteration, chosing the all points which belong to kth
    %partition, and then, to calculate kth centroids.
    index = find(Xy(:,n+1)==k);
    temp = X(index,:);
    s = sum(temp);
    centroids(k,:) = s./size(index,1);
end

end

