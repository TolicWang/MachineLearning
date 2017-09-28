load('ex7data2.mat');

[m,n] = size(X);

K = 3;

centroids = zeros(K,n);
centroids = kMeansInitCentroids(X,K);

iterations = 10;
idx = zeros(m,1);
J_history = zeros(iterations,1);
for i = 1: iterations;
    
idx = findClosestCentroids(X,centroids);
J_history(i) = costFunction(X,idx,centroids,K);

centroids = computeCentroids(X,idx,K)
i
end;

figure(1);

plot(J_history);
xlabel('iterations');
ylabel('costFuntion J');




