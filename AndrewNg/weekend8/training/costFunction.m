function J = costFunction(X,idx,centroids,K)
J = 0;
[m,n] = size(X);
Xy = [X idx];
for i = 1:K;
index = find(Xy(:,n+1)==i);
temp = X(index,:);
subs = temp - centroids(i,:);
s = sum(sum(subs.^2));
J = J+s;
end

%J = J/100;


end

