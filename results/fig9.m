feq = [345600, 652800, 960000, 1113600, 1420800, 1574400, 1881600]
sp = [1750]
rp = [1975, 2539, 2961, 3159, 4178, 4831, 6080]
p =[225, 789, 1211, 1409, 2428, 3081, 4330]
t = [1232, 590, 517, 424, 326, 285, 248.]

plot(feq, rp, '-+',feq, p, '-+', 'LineWidth', 3, 'MarkerSize', 8)
hold on
yline(1750, '-g.', 'LineWidth', 3)
legend("$$Busy$$","$$Inference$$","$$Idle$$",'fontsize',18, 'Interpreter', "latex")
ylabel("")
xlabel("CPU Feq (Hz)")
ylim([0, 7000])
xlim([345600, 1881600])
ylabel("Power (mW)")

grid on
box on
set(gca,'fontname','aakar') 
ax = gca; 
ax.FontSize = 18; 

set(gcf,'Position',[500 300 800 347])
