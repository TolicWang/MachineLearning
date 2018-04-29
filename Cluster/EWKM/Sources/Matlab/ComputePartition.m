function  idx  = ComputePartition( X,lambda,centroids )
[m,n] = size(X);
idx = zeros(m,1);
 for i = 1:m

    % in each iteration, computing the weight-distance from all centroids to 
    % only one point, and chose the minimu one's index as the value of
    % partition.
    subs = centroids - X(i,:);
    distance2 = subs.^2;
    w_distance2 = distance2 .* lambda;
    [temp,idx(i)] =min(sum(w_distance2,2)); 
end
end

