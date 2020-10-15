clear all;
clc;
%repeat the "testing_the_MLP_predictions_with_new_data" for the specs:
%Main Memory : 8000KB - 64000KB
%Cache : 32KB -  512KB

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
X = linspace(8000,64000,32);
Y = linspace(32,512,32);
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
zlabel('Απόδοση');
xlabel('Κεντρική μνήμη');
ylabel('Μνήμη Cache');


