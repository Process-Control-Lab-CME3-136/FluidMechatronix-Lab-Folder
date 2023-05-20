% Load training data
train_data = load('0SysID_trainingData.mat');
test_data = load('0SysID_testingData.mat');


time = train_data.out.MotorSpeed.time;
pressure = train_data.out.Pressure.signals.values;
rpm = train_data.out.MotorSpeed.signals.values;

t_test = test_data.out.MotorSpeed.time;
t_pressure = test_data.out.Pressure.signals.values;
t_rpm = test_data.out.MotorSpeed.signals.values;


% normalize data
p_norm = double(normalize(pressure, 'range'));
rpm_norm = normalize(rpm, 'range');

t_p_norm = double(normalize(t_pressure, 'range'));
t_rpm_norm = normalize(t_rpm, 'range');

Ts = 0.2;

train_data = iddata(p_norm, rpm_norm, Ts);
test_data = iddata(t_p_norm, t_rpm_norm, Ts);

%% State Space Model

nx = 2;

opt = ssestOptions();
opt.SearchOptions.MaxIterations = 20;
opt.Focus = 'simulation';   
opt.Display = 'off';

ss_model = ssest(train_data, nx,'Ts', Ts, opt);
compare(test_data, ss_model);

%% FOPTD model

% estimate I/O delay
delay = delayest(train_data, 1, 0, 1, 20);
opt = tfestOptions();
opt.InitializeMethod = 'all';
opt.InitialCondition = 'estimate';
opt.SearchMethod = 'auto';

opt.SearchOptions.MaxIterations = 30;

foptd_model = tfest(train_data, 1, 0, opt);


compare(test_data, foptd_model);


%% SOPTD model
soptd_model = tfest(train_data, 2, 0, opt);
compare(train_data, soptd_model);


%% Second order TF model

% np = 2, nz = 1, delay = 0.15

tf_model = tfest(train_data, 2, 1, opt);
compare(test_data, tf_model, foptd_model, soptd_model);


%% Linear ARX model

names = [train_data.OutputName, train_data.InputName];

opt = arxOptions();
opt.Focus = 'simulation';
opt.Display = 'off';

% na = 5, nb = 5, nk = 0
arx_model = arx(train_data, [5 5 0], opt);

compare(test_data, arx_model);

%% ARX model + Offset

opt = nlarxOptions();
opt.Focus = 'prediction';
opt.Display = 'off';

L = linearRegressor(names, {1:5, 0:5});
nlarx_model1 = nlarx(train_data, L, idLinear, opt);

compare(test_data, nlarx_model1)



%%  Non-Linear ARX Model


opt = nlarxOptions;
opt.focus = 'prediction';
opt.SearchMethod = 'gn';
opt.SearchOptions.MaxIterations = 10;

L = [linearRegressor(names, {1:5, 0:2}),customRegressor('u1', {0:10}, @(y)y.*y)];
nl_regressors = idnlarx(train_data.OutputName, train_data.InputName, L);

nlarx_model2 = nlarx(train_data, nl_regressors, opt);
compare(test_data, nlarx_model2);



%% Filter Transient response from data
   
% remove first 2 seconds of data, noise filtering 
data_size = length(time);
no_sp = floor(data_size/50);
p_norm_new = zeros(data_size-(no_sp+1)*10, 1);
rpm_norm_new = zeros(length(p_norm_new), 1);
time_new = linspace(1,data_size,data_size);



for i = 1:no_sp
    for p = 1:40
    p_norm_new(40*(i-1)+p) = p_norm(10+50*(i-1)+p);
    end
end
pump_freq_new = [];
for i = 1:no_sp
    for p = 1:40
    rpm_norm_new(40*(i-1)+p) = rpm_norm(10+50*(i-1)+p);
    end
end

data_size = length(t_p_norm);
no_sp = floor(data_size/50);
t_p_new = t_p_norm(1:data_size-(no_sp+1)*10);
t_rpm_new = t_rpm_norm(1:length(t_p_new));

plot(rpm_norm, p_norm, '.', rpm_norm_new, p_norm_new, '.')
xlabel(" RPM (Hz)")
ylabel("Pressure (kPa)")
legend("Raw", "Filtered")

%% Predicting output based on First Principles Model

x = rpm_norm_new;
y = p_norm_new;
y_predict = predict(x);

plot(x, y, '.', x, y_predict, '.');
ylabel('y predicted v/s actual (kPa)')
xlabel('RPM (Hz)')
sgtitle('Output Pressure v/s RPM model')
legend('Predicted', 'Actual')

actual = iddata(y, x, 'Ts', Ts);
predicted = iddata(y_predict, x, 'Ts', Ts);
test_actual = iddata(t_p_new, t_rpm_new, 'Ts', Ts);
test_predicted = iddata(predict(t_rpm_new), t_rpm_new, 'Ts', Ts);

figure(2)
compare(test_actual, test_predicted)

%% Residual Analysis For First Principle Model

residuals = zeros(length(test_data.y), 1);
cross_corr = zeros(length(test_data.y), 1);
for i = 1:length(test_data.y)

    residuals(i) = (predict(test_data.u(i)) - test_data.y(i))^2;
    cross_corr(i) = (predict(test_data.u(i))*test_data.u(i));
end

plot(1:length(residuals), residuals, '.--')
figure(2)
plot(1:length(residuals), cross_corr/(length(test_data.y)*var(test_data.u)*var(test_data.y)))


%% Static Second Order First Principle Model 

function y_predict = predict(x)

    y_predict = zeros(length(x), 1);
    % Define parameters for second-order equation: y(k) = a*x(k)^2+b*x(k)+c
    % Parameters obtained from curve fitting

    a = 0.9662;
    b = -0.004036;
    c = -0.001339;
    
    for i = 1:length(x)
        y_predict(i) = a*x(i)^2 + b*x(i)+c;
    end
    
end



