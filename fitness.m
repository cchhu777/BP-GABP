function error = fitness(x)
%�ú�������������Ӧ��ֵ

load data inputnum hiddennum_best outputnum net inputn outputn inputn_test outputps output_test

hiddennum=hiddennum_best;
%��ȡ
% ����������ӣ���֤ÿ�����н��һ��
setdemorandstream(pi);

w1=x(1:inputnum*hiddennum);%ȡ������������������ӵ�Ȩֵ
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);%��������Ԫ��ֵ
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);%ȡ������������������ӵ�Ȩֵ
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);%�������Ԫ��ֵ

net.trainParam.showWindow=0;  %���ط������

%����Ȩֵ��ֵ
net.iw{1,1}=reshape(w1,hiddennum,inputnum);%��w1��1��inputnum*hiddennum��תΪhiddennum��inputnum�еĶ�ά����
net.lw{2,1}=reshape(w2,outputnum,hiddennum);%���ľ���ı����ʽ
net.b{1}=reshape(B1,hiddennum,1);%1��hiddennum�У�Ϊ���������Ԫ��ֵ
net.b{2}=reshape(B2,outputnum,1);

%����ѵ��
net=train(net,inputn,outputn);
% 20221017�ش��޸ģ�����inputn_test�����Ϊ��inputn
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=mse(output_test,test_simu);
%  an=sim(net,inputn);
%  n_simu=mapminmax('reverse',an,outputps);
%  output=mapminmax('reverse',outputn,outputps);
%  
% error=mse(output,n_simu);



