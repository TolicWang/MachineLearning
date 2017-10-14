function idx = findClosestCentroids(X, centroids,weight)

K = size(centroids, 1);
idx = zeros(size(X,1), 1);

m = size(X,1);

for i = 1:m;
    
    subs = centroids - X(i,:);
    
    distance2 = sum((subs.^2).*weight,2);
    
    %distance2 = sum((subs.^2),2); % without weight
    
    [temp,idx(i)] = min(distance2);
    
end

end

