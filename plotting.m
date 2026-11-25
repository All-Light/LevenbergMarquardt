


% 2, example func y(t) = x_1 e^(x_2 t)
function [F] = getObjectiveFunction(x)
    ti = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    yi = [7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15];
    F = (x(1)*exp(x(2)*ti) - yi).';
end


function [err] = totalErr(x)
    err = sum(getObjectiveFunction(x));
end


function plot_func(x)
    ti = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    yi = [7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15];

    t = linspace(0,5,100);
    y = x(1)*exp(x(2)*t);

    scatter(ti, yi, 'filled')
    hold on
    plot(t, y, '-r', 'LineWidth',1.5)
    hold off
    xlabel('t')
    ylabel('y')
    legend('Given data', 'Fitted curve')

end

function plt_comparison(x,x_opt)
    ti = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    yi = [7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15];

    t = linspace(0,5,100);
    y = x(1)*exp(x(2)*t);
    y_opt = x_opt(1)*exp(x_opt(2)*t);

    scatter(ti, yi, 'filled')
    hold on
    plot(t, y, '-r', 'LineWidth',1.5, 'Color','r')
    plot(t, y_opt, '-r', 'LineWidth',1.5, 'Color','b')
    hold off
    xlabel('t')
    ylabel('y')
    legend('Given data', 'Fitted curve (LM)', 'Fitted curve (LSQCurveFit)')

end
ti = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
yi = [7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15];
F = @(x,t) x(1) * exp(x(2)*t);

x0 = [-125,2.2]; %starting guess

[x,resnorm,residual]  = levmarq(@getObjectiveFunction, x0);

options = optimset('Display', 'iter', 'Algorithm', 'levenberg-marquardt');
% matlab built in solver:
x_opt = lsqnonlin(@getObjectiveFunction,x0, [],[],options);

%plot_func(x)
sum(abs(x-x_opt))
plt_comparison(x,x_opt)

%getObjectiveFunction(x0)
%plot_func(x0)