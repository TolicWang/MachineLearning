function J = costFunction(X,idx,centroid,K,Weight)
J = 0;

[ m , n] = size(X);
Xy = [X idx];
D= zeros(1,n);
for i = 1: K;
    index = find(Xy(:,n+1)==i);
    temp = X(index,:);
    square = (temp-centroid(i,:)).^2;
    D = D + sum(square);
end


J = sum(Weight.*D);


end

