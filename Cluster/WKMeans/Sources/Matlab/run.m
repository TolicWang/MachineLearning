 %function score =  run(belta)   % turn on, if find beta

temp = load('4k2_far.txt');
%load('noise.mat');
%X = [temp(:,2:end),noise];
X = [temp(:,2:end)];
label = temp(:,1);
[m,n] = size(X);
K = 4;

centroids = kMeansInitCentroids(X,K);
r = rand(1,n);
weight = r./sum(r);

belta = 2; % turn off, if find beta

iterations = 30;
idx = zeros(m,1);

J = zeros(iterations,1);


for i = 1: iterations;
    
idx = findClosestCentroids(X,centroids,weight);%   find u

J(i) = costFunction(X,idx,centroids,K,weight);

centroids = computeCentroids(X,idx,K);% find  z

weight=Weight( X,centroids,idx,K,belta);  % find w
end

score = NMI(idx,label);


%idx1 = kmeans(X,K);
%score1 = NMI(idx1,label);



%fprintf("socre of K-means= %f.\n",score1);
fprintf("socre of K-W-means= %f.\n\n",score);


%% In practice, we need to run the W-k-means algorithm several times on the
%% same data set with different intial centroids and initial weights.
%%  After several times, the score of K-w-means will be about to 1,but the K-means
%% always about 0.47.

 
%  figure(1);
%  hold on;
% plot(J,'r');
% xlabel('iterations');
% ylabel('costFuntion J');



