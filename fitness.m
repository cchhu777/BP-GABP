function error = fitness(x)
%该函数用来计算适应度值

load data inputnum hiddennum_best outputnum net inputn outputn inputn_test outputps output_test

hiddennum=hiddennum_best;
%提取
% 设置随机种子，保证每次运行结果一样
setdemorandstream(pi);

w1=x(1:inputnum*hiddennum);%取到输入层与隐含层连接的权值
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);%隐含层神经元阈值
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);%取到隐含层与输出层连接的权值
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);%输出层神经元阈值

net.trainParam.showWindow=0;  %隐藏仿真界面

%网络权值赋值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);%将w1由1行inputnum*hiddennum列转为hiddennum行inputnum列的二维矩阵
net.lw{2,1}=reshape(w2,outputnum,hiddennum);%更改矩阵的保存格式
net.b{1}=reshape(B1,hiddennum,1);%1行hiddennum列，为隐含层的神经元阈值
net.b{2}=reshape(B2,outputnum,1);

%网络训练
net=train(net,inputn,outputn);
% 20221017重大修改，由用inputn_test仿真改为用inputn
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=mse(output_test,test_simu);
%  an=sim(net,inputn);
%  n_simu=mapminmax('reverse',an,outputps);
%  output=mapminmax('reverse',outputn,outputps);
%  
% error=mse(output,n_simu);



