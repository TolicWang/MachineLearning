function [ X,y] = PreProcessing
temp = load('./DataSet/wine.txt');

 
  y = temp(:,1);
   X = temp(:,2:end);




end

