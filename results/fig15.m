e=[0.7976,  1.9805,  2.5474, 2.9498, 3.5547]
t=[131.1, 277.2, 290.8, 317.1, 343.4]

e1= [1.435, 2.5944, 3.753, 4.9165, 6.6499]
t1= [106.8, 328.8, 435.2, 551.6, 718.1]

% 30， 23，32， 40，46
% 52 42 33 15
vv = [10, 18, 26, 34, 46]

subplot(1,2,1)
plot(vv, e, '-b|',vv, e1, '-r|', 'LineWidth',3, 'MarkerSize',12)
title("(a)")
ylabel("Energy Consumption (J)")
xlabel("Number of IoT Devices")
legend("P","R", 'fontsize',18, 'Interpreter', "latex")
xlim([5,50])

grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18; 

subplot(1,2,2)
plot(vv, t, '-b|',vv, t1, '-r|', 'LineWidth',3, 'MarkerSize',12)
title("(b)")
ylabel("Latency (ms)")
xlabel("Number of IoT Devices")
legend("P","R", 'fontsize',18, 'Interpreter', "latex")
xlim([5,50])

grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18; 

set(gcf,'Position',[500 300 800 347])
