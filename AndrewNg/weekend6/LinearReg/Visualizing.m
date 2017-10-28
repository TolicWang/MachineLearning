function Visualizing(X,y)
m = size(X,1);
plot(X,y,'rx','MarkerSize',10,'LineWidth',1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
end