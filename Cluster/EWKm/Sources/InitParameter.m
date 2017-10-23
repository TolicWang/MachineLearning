function [ lambda,K,gamma ] = InitParameter( X)

gamma = 1;% initialize parameter gamma
K = 3;
[m,n] = size(X); % m ,n represent the number of dimensions and points respectively;         
lambda = zeros(K,n)+1/n; % initialize all weights to 1/n

end

