function [ X,y] = PreProcessing
temp = load('DataSet\leuk72_3k.txt');

 
  y = temp(:,1);
   X = temp(:,2:end);




end

