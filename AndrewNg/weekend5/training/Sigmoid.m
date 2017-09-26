function [ sigmoid ] = Sigmoid( z )
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

demension = size(z);
sigmoid = zeros(demension);
sigmoid = 1.0 ./ (1+exp(-z));




end

