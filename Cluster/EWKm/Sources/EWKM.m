
[X,y] = PreProcessing;% preProcessing data
[lambda,K,gamma] = InitParameter(X);
[m,n] = size(X);
iterations = 25;
centroids = InitCentroids(X,K);
for i = 1:iterations
    idx = ComputePartition(X,lambda,centroids);% fixed Z,lambda to get W
    centroids = ComputeCentroids(X,idx,K);%fixed W,lambda to get z
    lambda = ComputeWeight(X,idx,gamma,centroids);% fixed W,z to get lambda
 end

fprintf("NMI = %f\n",NMI(idx,y));% After several runing, it maybe reach up to about 0.91


