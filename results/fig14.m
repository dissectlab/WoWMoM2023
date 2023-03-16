vv = [0.005, 0.05, 0.5, 1, 2,3,4, 5, 8]
e = [1.3313, 1.2422, 1.1856, 1.1491, 1.1181, 1.103, 1.081 ,1.059, 1.03]
t = [256.3, 260.1, 261.2, 263.4, 266.7, 269.5, 272.3, 274.4, 280.5]

plot(vv, e, '-b+', 'LineWidth',2, 'MarkerSize',8)
ylabel("Energy Consumption (J)")
xlabel("V")
%xlim([1, 20])
%ylim([0.25, 0.4])
%legend("AlexNet",'fontsize',18, 'Interpreter', "latex")
yyaxis right
plot(vv, t, '-r+', 'LineWidth',2, 'MarkerSize',8)
ylabel("Latency (ms)")

grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18; 

set(gcf,'Position',[500 300 800 347])
