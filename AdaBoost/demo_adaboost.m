clc;
clear;
load iris; 

T                       = 50;

options.weaklearner     = 0;
options.epsi            = 0.5;
options.lambda          = 1e-3;
options.max_ite         = 3000;

model                   = gentleboost_model(X , y , T , options);
[yest , fx]             = gentleboost_predict(X , model , options);
Perf                    = sum(y == yest)/length(y)

plot(fx')