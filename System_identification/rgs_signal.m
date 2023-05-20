% RGS signal generator
seed = 10;
rng(seed);

NumChannel = 1;
NumPeriod = 1;
NumSamples = 45000;
SP_Samples = 50;
Period = NumSamples/SP_Samples;
MaxSP = 50;
MinSP = 10;
Ts = 0.2;


rgs_raw = idinput([Period, NumChannel, NumPeriod], 'rgs', [0, 1], [MinSP, MaxSP]);
rgs_new = zeros(NumSamples, 1);
% for each value, take 50 samples
for i=1:Period
    if rgs_raw(i) >= MaxSP
        rgs_raw(i) = MaxSP;
    elseif rgs_raw(i) <= MinSP
        rgs_raw(i) = MinSP;
    end

    for j=1:SP_Samples
        rgs_new(50*(i-1)+j) = rgs_raw(i);
    end
end

u = iddata([], rgs_new, Ts);
plot(u)

save('0SysID_SignalData', 'u');