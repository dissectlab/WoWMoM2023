y = [
[0.1158,0.4707, 0.4134];
[0.2436,0.5015, 0.2547];
[0.3213,0.5025, 0.1761];
[0.3585,0.5018, 0.1400]
%[0.3820,0.5052, 0.1122]
]
subplot(1,3,1)
bar(y, 'stacked')
ylabel('Offloading Decisions','fontsize',18)
xlabel('Data Rate (Mbps)','fontsize',18)
title("(a)")
ylim([0,1])
legend("P", "R", "L", 'fontsize',18, 'Interpreter', "latex")
set(gca,'XTickLabel',{'80','120', "160", "200"});
grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18;

e1 = [2.4311, 1.718,  1.4257, 1.2444, 1.1211]
e2 = [3.5995, 2.4012, 1.8015, 1.443,  1.2035]
y =[
    [2.4311,3.5995,6.462];
    [1.718,2.4012,6.462];
    [1.4257,1.8015,6.462];
    [1.2444,1.443,6.462]
]
subplot(1,3,2)
bar(y)
ylabel('Energy Consumption (J)','fontsize',18)
xlabel('Data Rate (Mbps)','fontsize',18)
title("(b)")
ylim([0,7])
legend("P", "R", "L", 'fontsize',18, 'Interpreter', "latex")
set(gca,'XTickLabel',{'80','120', "160", "200","240"});
grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18;

y =[
    [258.0, 326.2, 480.979]; %21
    [257.0, 302.2, 480.979]; %15 
    [259.0, 290.2, 480.979]; %10
    [260.0, 283.2, 480.979]  %8
]
subplot(1,3,3)
bar(y)
ylabel('Latency (ms)','fontsize',18)
xlabel('Data Rate (Mbps)','fontsize',18)
title("(c)")
ylim([0,550])
legend("P", "R", "L", 'fontsize',18, 'Interpreter', "latex")
set(gca,'XTickLabel',{'80','120', "160", "200","240"});
grid on
box on
set(gca,'fontname','times') 
ax = gca; 
ax.FontSize = 18;

set(gcf,'Position',[500 300 1200 347])
