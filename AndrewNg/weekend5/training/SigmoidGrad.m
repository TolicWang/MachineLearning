function [ grad ] = SigmoidGrad( z )
%SIGMOIDGRAD Summary of this function goes here
%   Detailed explanation goes here

demension = size(z);
grad = zeros(demension);
grad = Sigmoid(z).*(1-Sigmoid(z));

end

