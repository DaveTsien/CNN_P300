clear all; clc;
load('./Subject_A_Train_120Hz.mat','data_norm_train','label_train');
load('./Subject_A_Test_120Hz.mat','data_norm_test','label_test');
Nelec = 64;     
m = 0.2;
ckernelsize = 64;
cmaps = 10;
cneuron = 78;
cskernelsize = 13;
csmaps = 5;
csneuron = 6;
hiddenNeuron = 100;
learningRate1 = 2*m/(cneuron*sqrt(ckernelsize));   %change
learningRate2 = 2*m/(csneuron*sqrt(cskernelsize));
learningRate3 = 2*m/(sqrt(csneuron*cmaps*csmaps));
learningRate4 = 2*m/(sqrt(hiddenNeuron));
cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', cmaps, 'kernelsize', [1 ckernelsize], 'perNeural', cneuron, 'learningRate', learningRate1)     %convolution layer
    struct('type', 'cs', 'outputmaps', csmaps*cmaps, 'kernelsize', [cskernelsize 1], 'perNeural', csneuron, 'learningRate', learningRate2)     %convolution and subsampling layer
    struct('type', 'h', 'outputmaps', 1, 'perNeural', hiddenNeuron, 'learningRate', learningRate3)                  % hidden layer
};
cnn.learningRate = learningRate4;          %output layer
cnn.perNeural = 2;
cnn = P300_CNNsetup(cnn);
opts.trainepochs = 5100;     
opts.testepochs = 18000; 
opts_test.testepochs = 18000;


tp = find(label_train == 1);
tf = find(label_train == 0);
data_norm_train_T = data_norm_train(:,:,tp);
data_norm_train_F = data_norm_train(:,:,tf);
% rand_F
rand_F = randperm(12750,2550);
data_norm_train_F = data_norm_train_F(:,:,rand_F);
opts.rand = rand_F;
% data_trains = data_norm_train;
% label_trains = [abs(label_train-1)' label_train'];
data_trains = cat(3,data_norm_train_T,data_norm_train_F);
label_trains = [zeros(2550,1),ones(2550,1);ones(2550,1),zeros(2550,1)];
%rand_all
randp = randperm(5100,5100);
data_trains = data_trains(:,:,randp);
label_trains = label_trains(randp,:);
Ttrain = find(label_trains(:,2) == 1);

Ttest = find(label_test == 1);
label_test = [abs(label_test-1)' label_test'];
% load('../Result3/CNN_OPTS_SubjectA_train_ite_100_1122.mat')
time = clock;
tip = strcat(num2str(time(3)),num2str(time(4)),num2str(time(5)),num2str(fix(time(6))));

ers_train = zeros(1,300);
FNs_train = zeros(1,300);
ers_test = zeros(1,300);
FNs_test = zeros(1,300);
for epoch = 1:30     %iterator
    opts.er = 0;
    opts.erp = [];
    opts.output = [];
    cnn.L = 0;
    cnn.rL = [];
    cnn.rLs = [];
    
    cnn = P300_CNNtrain(cnn, data_trains, label_trains, opts);
    [opts, bad] = P300_CNNtest_train(cnn, data_trains, label_trains, opts);
    opts.FN = intersect(opts.erp,Ttrain);
    ers_train(epoch) = opts.er;
    FNs_train(epoch) = size(opts.FN,1);
    
    opts_test.er = 0;
    opts_test.erp = [];
    opts_test.output = [];
    opts_test.FN = [];
    [opts_test,~] = P300_CNNtest(cnn,data_norm_test,label_test,opts_test);
    opts_test.FN = intersect(Ttest,opts_test.erp');
    
    ers_test(epoch) = opts_test.er;
    FNs_test(epoch) = size(opts_test.FN,1);
    save(strcat('Results/CNN_OPTS_SubjectA_train_ite','_',num2str(epoch),'_',tip),'cnn','opts','opts_test');
end
save('Result2/test_200','ers_train','FNs_train','ers_test','FNs_test');