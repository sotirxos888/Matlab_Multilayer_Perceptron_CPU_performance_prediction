clear all;
clc;
%Using the best neural network from the "MLP build and test" compare
%the CPUs with the specs shown below:
%Duty cycle: 200ns                                            
%Main memory (min): 3000KB
%Channels(min):6
%Channels (max):16
%Main memory(max): 8000KB - 16000 KB
%Cache : 32KB - 128KB

%data input
x=xlsread('machine_CPU_NN','D2:I210');
y=xlsread('Machine_CPU_NN','J2:J210');
%data normalization
[xsc,ps_xsc]=mapminmax(x');
[ysc,ps_ysc]=mapminmax(y');
%create network
rand('seed',10);
net=feedforwardnet([17 4]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:90;
net.divideParam.valInd = 91:150;
net.divideParam.testInd = 151:209;
[net,tr]=train(net,xsc,ysc);
%neural network predictions
Ypsc = net(xsc);
%denormalization
Yp = mapminmax('reverse',Ypsc,ps_ysc)';
%3D graphic
X = linspace(8000,16000,16);
Y = linspace(32,128,16);
[xx,yy]=meshgrid(X,Y);
for i=1:length(X)
    for j=1:length(Y)
       x = [200 3000 xx(i,j) yy(i,j) 6 16];
       Xsc = mapminmax('apply',x',ps_xsc);
       ypsc = net(Xsc);
       yp(i,j) = mapminmax('reverse',ypsc,ps_ysc);
    end
end
    mesh(xx,yy,yp')
title('Benchmark Test');
zlabel('Performance');
xlabel('Main memory');
ylabel('Μνήμη Cache');


