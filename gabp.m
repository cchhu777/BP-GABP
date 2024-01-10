%% �˳���ΪBP������ͻ����Ŵ��㷨�Ż���BP������Ԥ��Ч���Ա�
% ��ջ�������
clear,close all
clc
warning off
for iii=1
error_index=zeros(10,342);  %���������ľ���
%% ��ȡ����
data=xlsread('train.xls',iii);
input=data(:,1:2);
ii=3
output=data(:,ii);
%% ����ѵ�����ݺ�Ԥ������
L=length(output);  %����������
num=999;         %����ѵ����������

input_train = input(1:num,:)';
output_train = output(1:num,:)';
input_test = input(num+1:end,:)';
output_test = output(num+1:end,:)';

disp(['ѵ����������',num2str(num)])
disp(['������������',num2str(L-num)])

%% ���ݹ�һ��
[inputn,inputps]=mapminmax(input_train);%��һ����[-1,1]֮�䣬inputps��������һ��ͬ���Ĺ�һ��
[outputn,outputps]=mapminmax(output_train);
inputn_test=mapminmax('apply',input_test,inputps);% �Բ����������ݽ��й�һ��

%% �ڵ����
inputnum=size(input_train,1);       %�����ڵ����
outputnum=size(output_train,1);      %�����ڵ����

%% ɸѡ���������ڵ����
%������ڵ��������鹫ʽp=sqrt(m+n)+a ��aΪ0~10������ȡ7~18��������

hiddennumll=1;    %����������ֵ
m=1;            %���������ӵĲ���
hiddennumuu=2;  %����������ֵ
MSEAll=[];     %��ʼ��mse

for hiddennum=hiddennumll:m:hiddennumuu
    
    net0=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm');% ���ݺ���ʹ��purelin�������ݶ��½���ѵ��
    
    %�����������
    net0.trainParam.epochs=1000;         % ѵ����������������Ϊ1000��
    net0.trainParam.lr=0.1;              % ѧϰ����
    net0.trainParam.goal=0.00001;        % ѵ��Ŀ����С���
    net0.trainParam.show=25;             % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
    net0.trainParam.mc=0.01;             % ��������
    net0.trainParam.min_grad=1e-6;       % ��С�����ݶ�
    net0.trainParam.max_fail=6;          % ���ʧ�ܴ���
    net0.trainParam.showWindow=0;        %����ʾѵ�����棬��ͼ
    
    %ѵ��
    [net0,tr]=train(net0,inputn,outputn);%��ʼѵ��������inputn,outputn�ֱ�Ϊ�����������
    
    %���ѵ�����ݹ�һ���ķ���ֵ
    an1=sim(net0,inputn);
    %����ѵ�����ľ������mse
    error1=an1-outputn;
    [rr,ss]=size(error1);
    mse1=error1*error1'/ss;
    %��¼ÿһ���������mse
    MSEAll=[MSEAll,mse1];
end

%���������ڵ���hiddennum_best
p=find(MSEAll==min(MSEAll));
hiddennum_best=(p-1)*m+hiddennumll;

%�����ͬ�������Ӧ��mseֵ
for countt=hiddennumll:m:hiddennumuu
    msee=MSEAll((countt-hiddennumll)/m+1);
    disp(['������ڵ���Ϊ',num2str(countt),'ʱ�ľ�������ǣ� ',num2str(msee)])
end
disp(' ')
disp(['���������ڵ���Ϊ��    ',num2str(hiddennum_best)])



%% ����BP������
disp(' ')
disp('��׼��BP�����磺')
net0=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

%�����������
net0.trainParam.epochs=1000;         % ѵ����������������Ϊ1000��
net0.trainParam.lr=0.01;                   % ѧϰ���ʣ���������Ϊ0.01
net0.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
net0.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
net0.trainParam.mc=0.01;                 % ��������
net0.trainParam.min_grad=1e-6;       % ��С�����ݶ�
net0.trainParam.max_fail=6;               % ���ʧ�ܴ���
net0.trainParam.showWindow=0;     %����ʾѵ�����棬��ͼ

%��ʼѵ��
[net0,tr0]=train(net0,inputn,outputn);
% figure
% plotperform(tr0)   %ѵ������������ߣ���ͼ
% Ԥ��
an0=sim(net0,inputn_test); %��ѵ���õ�ģ�ͽ��з���
an3 = sim(net0, inputn);  %ѵ����ģ����
%Ԥ��������һ����������
test_simu0=mapminmax('reverse',an0,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
train_simu0=mapminmax('reverse',an3,outputps);  % ѵ����ģ������ԭ
%���ָ��
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);
error_index(1,ii-2)=ii-2;
error_index(2,ii-2)=mae0;
error_index(3,ii-2)=mse0;
error_index(4,ii-2)=rmse0;
error_index(5,ii-2)=mape0;

%% �Ŵ��㷨Ѱ����Ȩֵ��ֵ
% disp(' ')
% disp('GA�Ż�BP�����磺')
% net=newff(inputn,outputn,hiddennum_best,{'tansig','purelin'},'trainlm');% ����ģ��

% %�����������
% net.trainParam.epochs=1000;         % ѵ����������������Ϊ1000��
% net.trainParam.lr=0.01;                   % ѧϰ���ʣ���������Ϊ0.01
% net.trainParam.goal=0.00001;                    % ѵ��Ŀ����С����������Ϊ0.0001
% net.trainParam.show=25;                % ��ʾƵ�ʣ���������Ϊÿѵ��25����ʾһ��
% net.trainParam.mc=0.01;                 % ��������
% net.trainParam.min_grad=1e-6;       % ��С�����ݶ�
% net.trainParam.max_fail=6;               % ���ʧ�ܴ���
% net.trainParam.showWindow=0;     %����ʾѵ�����棬��ͼ
% 
% save data inputnum hiddennum_best outputnum net inputn outputn  inputn_test outputps output_test
% 
% 
% %��ʼ��ga����
% PopulationSize_Data=30;   %��ʼ��Ⱥ��ģ30-40
% MaxGenerations_Data=200;   %����������
% CrossoverFraction_Data=0.8;  %�������
% MigrationFraction_Data=0.05;   %�������0.05-0.1
% nvars=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum;    %�Ա�������
% lb=repmat(-3,nvars,1);    %�Ա�������
% ub=repmat(3,nvars,1);   %�Ա�������
% 
% %�����Ŵ��㷨����
% options = optimoptions('ga');
% options = optimoptions(options,'PopulationSize', PopulationSize_Data);
% options = optimoptions(options,'CrossoverFraction', CrossoverFraction_Data);
% options = optimoptions(options,'MigrationFraction', MigrationFraction_Data);
% options = optimoptions(options,'MaxGenerations', MaxGenerations_Data);
% options = optimoptions(options,'SelectionFcn', @selectionroulette);   %���̶�ѡ��
% options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);   %���㽻��
% options = optimoptions(options,'MutationFcn', {  @mutationgaussian [] [] });   %��˹����
% options = optimoptions(options,'Display', 'off');    %��off��Ϊ����ʾ�������̣���iter��Ϊ��ʾ��������
% % options = optimoptions(options,'PlotFcn', { @gaplotbestf });    %�����Ӧ�ȣ���ͼ
% 
% %���
% [x,fval] = ga(@fitness,nvars,[],[],[],[],lb,ub,[],[],options);
% 
% 
% %%�����ų�ʼ��ֵȨֵ��������Ԥ��
% % %���Ŵ��㷨�Ż���BP�������ֵԤ��
% % W1= net. iw{1, 1}%����㵽�м���Ȩֵ
% % B1 = net.b{1}%�м������Ԫ��ֵ
% %
% % W2 = net.lw{2,1}%�м�㵽������Ȩֵ
% % B2 = net. b{2}%��������Ԫ��ֵ
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

%% �Ż����������ѵ��
% [net,tr]=train(net,inputn,outputn);%��ʼѵ��������inputn,outputn�ֱ�Ϊ�����������
% %  figure
% %  plotperform(tr)   %ѵ������������ߣ���ͼ
% 
% %% �Ż�������������
% an1=sim(net,inputn_test);
% test_simu1=mapminmax('reverse',an1,outputps); %�ѷ���õ������ݻ�ԭΪԭʼ��������
% %���ָ��
% [mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);
% R2=(L*sum(test_simu1.*output_test)-sum(test_simu1)*sum(output_test))^2/((L*sum((test_simu1).^2)-(sum(test_simu1))^2)*(L*sum((output_test).^2)-(sum(output_test))^2)); 
% error_index(6,ii-4)=mae1;
% error_index(7,ii-4)=mse1;
% error_index(8,ii-4)=rmse1;
% error_index(9,ii-4)=mape1;
% error_index(10,ii-4)=R2;

 %% ��ͼ
%  figure
%  plot(1:L-num,output_test,'b-*')
%  hold on
%  plot(1:L-num,test_simu0,'rp-')
%  hold on
%  plot(1:L-num,test_simu1,'mo-')
%  legend('��ʵֵ','��׼BPԤ��ֵ','GABPԤ��ֵ')
%  xlabel('��������')
%  ylabel('ֵ')
%  title('������Ԥ��ֵ����ʵֵ�Ա�ͼ')
%  
%  figure
%  plot(1:L-num,error0,'rp-')
%  hold on
%  plot(1:L-num,error1,'mo-')
%  legend('��׼BPԤ�����','GABPԤ�����')
%  xlabel('��������')
%  ylabel('���')
%  title('������Ԥ��ֵ����ʵֵ���Ա�ͼ')
% %% ����д��
%  output_test_simu0=strcat('A',num2str(ii-4));
%  output_test_simu1=strcat('A',num2str(ii-4));
% % test simu�Ǵ��У����Ԥ��ֵ�ı�simu0δ�Ż���simu1�Ż���
%  xlswrite('test simu(1).xlsx',test_simu0,2*iii-1,output_test_simu0);
%  xlswrite('test simu(1).xlsx',test_simu1,2*iii,output_test_simu1);
% 
% 
% error_index=error_index';
% xlswrite('C:\Users\lenovo\Desktop\error(3).xls',error_index,iii);
end