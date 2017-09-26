function [ Theta ] = InitialTheta( row,column )
%INITIALTHETA Summary of this function goes here
%   Detailed explanation goes here


Theta = zeros(row,column);
init_epsilong = 0.12;
Theta = rand(row,column)*2*init_epsilong-init_epsilong;
end

