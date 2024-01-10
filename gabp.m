%% 此程序为BP神经网络和基于遗传算法优化的BP神经网络预测效果对比
% 清空环境变量
clear,close all
clc
warning off
for iii=1
error_index=zeros(10,342);  %存放误差结果的矩阵
%% 读取数据
data=xlsread('train.xls',iii);
input=data(:,1:2);
ii=3
output=data(:,ii);
%% 设置训练数据和预测数据
L=length(output);  %总样本个数
num=999;         %设置训练样本个数

input_train = input(1:num,:)';
output_train = output(1:num,:)';
input_test = input(num+1:end,:)';
output_test = output(num+1:end,:)';

disp(['训练样本数：',num2str(num)])
disp(['测试样本数：',num2str(L-num)])

%% 数据归一化
[inputn,inputps]=mapminmax(input_train);%归一化到[-1,1]之间，inputps用来作下一次同样的归一化
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);% 对测试样本数据进行归一化

%% 节点个数
inputnum=size(input_train,1);       %输入层节点个数
outputnum=size(output_train,1);      %输出层节点个数

%% 筛选最佳隐含层节点个数
%隐含层节点数量经验公式p=sqrt(m+n)+a ，a为0~10；这里取7~18进行试验

hiddennumll=1;    %隐含层下限值
m=1;            %隐含层增加的步距
hiddennumuu=2;  %隐含层上限值
MSEAll=[];     %初始化mse

for hiddennum=hiddennumll:m:hiddennumuu
    
    net0=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm');% 传递函数使用purelin，采用梯度下降法训练
    
    %网络参数配置
    net0.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
    net0.trainParam.lr=0.1;              % 学习速率
    net0.trainParam.goal=0.00001;        % 训练目标最小误差
    net0.trainParam.show=25;             % 显示频率，这里设置为每训练25次显示一次
    net0.trainParam.mc=0.01;             % 动量因子
    net0.trainParam.min_grad=1e-6;       % 最小性能梯度
    net0.trainParam.max_fail=6;          % 最高失败次数
    net0.trainParam.showWindow=0;        %不显示训练界面，作图
    
    %训练
    [net0,tr]=train(net0,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本
    
    %获得训练数据归一化的仿真值
    an1=sim(net0,inputn);
    %计算训练集的均方误差mse
    error1=an1-outputn;
    [rr,ss]=size(error1);
    mse1=error1*error1'/ss;
    %记录每一个隐含层的mse
    MSEAll=[MSEAll,mse1];
end

%最佳隐含层节点数hiddennum_best
p=find(MSEAll==min(MSEAll));
hiddennum_best=(p-1)*m+hiddennumll;

%输出不同隐含层对应的mse值
for countt=hiddennumll:m:hiddennumuu
    msee=MSEAll((countt-hiddennumll)/m+1);
    disp(['隐含层节点数为',num2str(countt),'时的均方误差是： ',num2str(msee)])
end
disp(' ')
disp(['最佳隐含层节点数为：    ',num2str(hiddennum_best)])



%% 构建BP神经网络
disp(' ')
disp('标准的BP神经网络：')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

%网络参数配置
net0.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net0.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net0.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
net0.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
net0.trainParam.mc=0.01;                 % 动量因子
net0.trainParam.min_grad=1e-6;       % 最小性能梯度
net0.trainParam.max_fail=6;               % 最高失败次数
net0.trainParam.showWindow=0;     %不显示训练界面，作图

%开始训练
[net0,tr0]=train(net0,inputn,outputn);
% figure
% plotperform(tr0)   %训练集的误差曲线，作图
% 预测
an0=sim(net0,inputn_test); %用训练好的模型进行仿真
an3 = sim(net0, inputn);  %训练集模拟结果
%预测结果反归一化与误差计算
test_simu0=mapminmax('reverse',an0,outputps); %把仿真得到的数据还原为原始的数量级
train_simu0=mapminmax('reverse',an3,outputps);  % 训练集模拟结果还原
%误差指标
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);
error_index(1,ii-2)=ii-2;
error_index(2,ii-2)=mae0;
error_index(3,ii-2)=mse0;
error_index(4,ii-2)=rmse0;
error_index(5,ii-2)=mape0;

%% 遗传算法寻最优权值阈值
% disp(' ')
% disp('GA优化BP神经网络：')
% net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% 建立模型

% %网络参数配置
% net.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
% net.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
% net.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.0001
% net.trainParam.show=25;                % 显示频率，这里设置为每训练25次显示一次
% net.trainParam.mc=0.01;                 % 动量因子
% net.trainParam.min_grad=1e-6;       % 最小性能梯度
% net.trainParam.max_fail=6;               % 最高失败次数
% net.trainParam.showWindow=0;     %不显示训练界面，作图
% 
% save data inputnum hiddennum_best outputnum net inputn outputn  inputn_test outputps output_test
% 
% 
% %初始化ga参数
% PopulationSize_Data=30;   %初始种群规模30-40
% MaxGenerations_Data=200;   %最大进化代数
% CrossoverFraction_Data=0.8;  %交叉概率
% MigrationFraction_Data=0.05;   %变异概率0.05-0.1
% nvars=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;    %自变量个数
% lb=repmat(-3,nvars,1);    %自变量下限
% ub=repmat(3,nvars,1);   %自变量上限
% 
% %调用遗传算法函数
% options = optimoptions('ga');
% options = optimoptions(options,'PopulationSize', PopulationSize_Data);
% options = optimoptions(options,'CrossoverFraction', CrossoverFraction_Data);
% options = optimoptions(options,'MigrationFraction', MigrationFraction_Data);
% options = optimoptions(options,'MaxGenerations', MaxGenerations_Data);
% options = optimoptions(options,'SelectionFcn', @selectionroulette);   %轮盘赌选择
% options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);   %两点交叉
% options = optimoptions(options,'MutationFcn', {  @mutationgaussian [] [] });   %高斯变异
% options = optimoptions(options,'Display', 'off');    %‘off’为不显示迭代过程，‘iter’为显示迭代过程
% % options = optimoptions(options,'PlotFcn', { @gaplotbestf });    %最佳适应度，作图
% 
% %求解
% [x,fval] = ga(@fitness,nvars,[],[],[],[],lb,ub,[],[],options);
% 
% 
% %%把最优初始阀值权值赋予网络预测
% % %用遗传算法优化的BP网络进行值预测
% % W1= net. iw{1, 1}%输入层到中间层的权值
% % B1 = net.b{1}%中间各层神经元阈值
% %
% % W2 = net.lw{2,1}%中间层到输出层的权值
% % B2 = net. b{2}%输出层各神经元阈值
% setdemorandstream(pi);
% 
% w1=x(1:inputnum*hiddennum_best);
% B1=x(inputnum*hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best);
% w2=x(inputnum*hiddennum_best+hiddennum_best+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum);
% B2=x(inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+1:inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum);
% 
% net.iw{1,1}=reshape(w1,hiddennum_best,inputnum);
% net.lw{2,1}=reshape(w2,outputnum,hiddennum_best);
% net.b{1}=reshape(B1,hiddennum_best,1);
% net.b{2}=reshape(B2,outputnum,1);

%% 优化后的神经网络训练
% [net,tr]=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本
% %  figure
% %  plotperform(tr)   %训练集的误差曲线，作图
% 
% %% 优化后的神经网络测试
% an1=sim(net,inputn_test);
% test_simu1=mapminmax('reverse',an1,outputps); %把仿真得到的数据还原为原始的数量级
% %误差指标
% [mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);
% R2=(L*sum(test_simu1.*output_test)-sum(test_simu1)*sum(output_test))^2/((L*sum((test_simu1).^2)-(sum(test_simu1))^2)*(L*sum((output_test).^2)-(sum(output_test))^2)); 
% error_index(6,ii-4)=mae1;
% error_index(7,ii-4)=mse1;
% error_index(8,ii-4)=rmse1;
% error_index(9,ii-4)=mape1;
% error_index(10,ii-4)=R2;

 %% 作图
%  figure
%  plot(1:L-num,output_test,'b-*')
%  hold on
%  plot(1:L-num,test_simu0,'rp-')
%  hold on
%  plot(1:L-num,test_simu1,'mo-')
%  legend('真实值','标准BP预测值','GABP预测值')
%  xlabel('测试样本')
%  ylabel('值')
%  title('神经网络预测值和真实值对比图')
%  
%  figure
%  plot(1:L-num,error0,'rp-')
%  hold on
%  plot(1:L-num,error1,'mo-')
%  legend('标准BP预测误差','GABP预测误差')
%  xlabel('测试样本')
%  ylabel('误差')
%  title('神经网络预测值和真实值误差对比图')
% %% 批量写入
%  output_test_simu0=strcat('A',num2str(ii-4));
%  output_test_simu1=strcat('A',num2str(ii-4));
% % test simu是存放校核年预报值的表，simu0未优化，simu1优化后
%  xlswrite('test simu(1).xlsx',test_simu0,2*iii-1,output_test_simu0);
%  xlswrite('test simu(1).xlsx',test_simu1,2*iii,output_test_simu1);
% 
% 
% error_index=error_index';
% xlswrite('C:\Users\lenovo\Desktop\error(3).xls',error_index,iii);
end