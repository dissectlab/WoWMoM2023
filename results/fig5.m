data =  [0.5744, 0.7387, 0.1782, 0.5341, 0.124, 0.2477, 0.1652, 0.0353, 0.0158, 0.0158, 0.004]
layer_latency =  [0, 27.2579, 10.1612, 21.7888, 4.3666, 34.2211, 22.0609, 33.0553, 121.3356, 103.923, 24.8342]

x=1:1:21

subplot(1,4,1)
y = [[0.0102, 648.6486, 29.0507];[47.6049, 834.9912, 27.892];[40.4949, 200.3565, 28.0988];[79.3638, 599.0856, 22.7438];[82.4224, 157.9804, 22.3614];[91.248, 276.9765, 18.9474];[131.8877, 182.0593, 15.3101];[169.7778, 69.3065, 12.5744];[296.7479, 15.0938, 4.4013];[337.7439, 15.2577, 1.071];[345.364, 0, 0]]
bar(y,'stacked')
ylabel('E2E Latency (ms)','fontsize',18)
xlabel('Cut Point','fontsize',18)
title("(a) 8 Mbps",'fontname','Times') 
legend("Local", "Network", "Remote", 'fontsize',18, 'Interpreter', "latex")
grid on
box on
set(gca,'fontname','Times') 
ax = gca; 
ax.FontSize = 18;
ax.YAxis.Exponent = 2;

subplot(1,4,2)
y = [[0.0062, 251.9516, 22.4112];[46.5253, 318.177, 20.7842];[58.0714, 77.1231, 20.3788];[91.4192, 231.4983, 18.3253];[106.0141, 59.4216, 16.4496];[131.1848, 107.5919, 13.4879];[151.7966, 71.9317, 11.9233];[174.1072, 14.9257, 8.4343];[284.7456, 7.6165, 3.6291];[344.7426, 7.4761, 1.6653];[358.9509, 0, 0]]
bar(y,'stacked')
ylabel('E2E Latency (ms)','fontsize',18)
xlabel('Cut Point','fontsize',18)
title("(b) 20 Mbps",'fontname','Times') 
legend("Local", "Network", "Remote", 'fontsize',18, 'Interpreter', "latex")
grid on
box on
set(gca,'fontname','Times') 
ax = gca; 
ax.FontSize = 18;
ax.YAxis.Exponent = 2;

subplot(1,4,3)
y = [[2.317638288e-05, 0.3243243];[0.13216056628383, 0.4174956];[0.11123515329621, 0.10017825000000001];[0.2249768891871, 0.2995428];[0.23158241509376, 0.0789902];[0.266581032, 0.13848825];[0.37454880244390004, 0.09102965];[0.4559828485725, 0.03465325];[0.82184330905, 0.0075469];[0.9560243004741, 0.0076288499999999995];[0.968421723204, 0.0]]
bar(y,'stacked')
ylabel('Energy Consumption (J)','fontsize',18)
xlabel('Cut Point','fontsize',18)
title("(c) 8 Mbps")
legend("Local", "Network", 'fontsize',18, 'Interpreter', "latex")
grid on
box on
set(gca,'fontname','Times') 
ax = gca; 
ax.FontSize = 18;
%ax.YAxis.Exponent = 2;

subplot(1,4,4)
y = [[3.806761035e-05, 0.14319995];[0.10595876012597, 0.1717469];[0.10061651539016, 0.09721465];[0.21314235871007997, 0.1261983];[0.21452288069988001, 0.08068295];[0.26221816849824, 0.07333015];[0.35875460263761, 0.06137475];[0.45154602285375, 0.036419];[0.7332508511249999, 0.0048525999999999994];[0.93582087439577, 0.00349045];[0.87539779995154, 0.0]]
bar(y,'stacked')
ylabel('Energy Consumption (J)','fontsize',18)
xlabel('Cut Point','fontsize',18)
title("(d) 20 Mbps")
legend("Local", "Network", 'fontsize',18, 'Interpreter', "latex")
grid on
box on
set(gca,'fontname','Times') 
ax = gca; 
ax.FontSize = 18; 
%ax.YAxis.Exponent = 2;

set(gcf,'Position',[500 300 1800 347])
