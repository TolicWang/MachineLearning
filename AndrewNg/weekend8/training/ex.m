%load('ex7data2.mat');

temp = load('DataSet\4k2_far.txt');
X = temp(:,2:end);
label = temp(:,1);

[m,n] = size(X);

K = 4;

centroids = zeros(K,n);
centroids = kMeansInitCentroids(X,K)
centroids = [8.4529 8.3170;3.5049 6.3118;7.2779 8.4457;8.4173 7.5257];
iterations = 10;
idx = zeros(m,1);

figure(1); hold on;

J_history = zeros(iterations,1);

previous = centroids ;
for i = 1: iterations;
    
idx = findClosestCentroids(X,centroids);

fprintf('Program paused. Press enter to continue.\n');
J_history(i) = costFunction(X,idx,centroids,K);

plotProgresskMeans(X, centroids, previous, idx, K, i);
previous = centroids ;
centroids = computeCentroids(X,idx,K);

pause;
end;


figure(2);

plot(J_history);
xlabel('iterations');
ylabel('costFuntion J');




