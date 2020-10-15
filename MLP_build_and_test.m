clear all
clc
%train a neural network to understand the connection between the CPU specs
%and the CPU performance

%data input
x=xlsread('machine_CPU_NN','D2:I210');
y=xlsread('Machine_CPU_NN','J2:J210');
%data normalization
[xsc,ps_xsc]=mapminmax(x');
[ysc,ps_ysc]=mapminmax(y');
%network creation
rand('seed',10);
net=feedforwardnet([5 3]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:90; %training data
net.divideParam.valInd = 91:150; %validation data
net.divideParam.testInd = 151:209; %test data
[net,tr]=train(net,xsc,ysc);
%neural network predictions
Ypsc = net(xsc);
%denormalization
Yp = mapminmax('reverse',Ypsc,ps_ysc)';
%calculate mare% and R^2
ym1 = sum(y(91:150))/60;
ym2 = sum(y(151:209))/59;
SSEval = sum((y(91:150)-Yp(91:150)).^2);
SSTval = sum((y(91:150)-ym1).^2);
SSEtest = sum((y(151:209)-Yp(151:209)).^2);
SSTtest = sum((y(151:209)-ym2).^2);
R2val = 1-(SSEval/SSTval);
R2test = 1-(SSEtest/SSTtest);
MAREval = 100*(sum(abs(y(91:150)-Yp(91:150))./y(91:150))/60);
MAREtest = 100*(sum(abs(y(151:209)-Yp(151:209))./y(151:209))/59);

%if the 1st layer has 5-20 neurons
%and the 2nd layer has 3-10 neurons
%train and save all the possibilities
for i=5:20
    for j=3:10
        rand('seed',10);
net=feedforwardnet([i j]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:90;
net.divideParam.valInd = 91:150;
net.divideParam.testInd = 151:209;
[net,tr]=train(net,xsc,ysc);
%neural network predictions
Ypsc = net(xsc);
%denormalization
Yp = mapminmax('reverse',Ypsc,ps_ysc)';
%calculate mare% and R^2
ym1 = sum(y(91:150))/60;
ym2 = sum(y(151:209))/59;
SSEval = sum((y(91:150)-Yp(91:150)).^2);
SSTval = sum((y(91:150)-ym1).^2);
SSEtest = sum((y(151:209)-Yp(151:209)).^2);
SSTtest = sum((y(151:209)-ym2).^2);
R2val_B(i,j) = 1-(SSEval/SSTval);
R2test_B(i,j) = 1-(SSEtest/SSTtest);
MAREval_B(i,j) = 100*(sum(abs(y(91:150)-Yp(91:150))./y(91:150))/60);
MAREtest_B(i,j) = 100*(sum(abs(y(151:209)-Yp(151:209))./y(151:209))/59);
    end
end