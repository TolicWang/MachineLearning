
temp = load('./DataSet/wine.txt');
y = temp(:,1);
X = temp(:,2:end);
[m,n] = size(X);
gamma = 1;% initialize parameter gamma
K = 3;   
lambda = zeros(K,n)+1/n;% initialize all weights to 1/n


iterations = 25;
centroids = InitCentroids(X,K);
for i = 1:iterations
    idx = ComputePartition(X,lambda,centroids);% fixed Z,lambda to get W
    centroids = ComputeCentroids(X,idx,K);%fixed W,lambda to get z
    lambda = ComputeWeight(X,idx,gamma,centroids);% fixed W,z to get lambda
 end

fprintf("NMI = %f\n",NMI(idx,y));% After several runing, it maybe reach up to about 0.91


