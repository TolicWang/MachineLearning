function [ weight ] = Weight( X,centroid,idx,K,belta )
%WEIGHT Summary of this function goes here
%   Detailed explanation goes here

[ m , n] = size(X);
Xy = [X idx];
D= zeros(1,n);
weight = zeros(1,n);
for i = 1: K;
    index = find(Xy(:,n+1)==i);
    temp = X(index,:);
    square = (temp-centroid(i,:)).^2;
    D = D + sum(square);
end



e = 1/(belta-1);
for j  = 1:n;
    temp = D(j)./D;
    weight(j) = 1/(sum(temp.^e));
end

