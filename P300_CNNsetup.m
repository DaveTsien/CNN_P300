function net = P300_CNNsetup(net)
%% Convolution layer
inputmaps = 1;
fan_out = net.layers{2}.outputmaps * net.layers{2}.kernelsize(2);
fan_in = inputmaps * net.layers{2}.kernelsize(2);
rng('shuffle');  %seed by time 
net.layers{2}.k = mat2cell((rand(net.layers{2}.outputmaps,net.layers{2}.kernelsize(2)) - 0.5)...
    * 2 * sqrt(6 / (fan_in + fan_out)),ones(1,net.layers{2}.outputmaps),net.layers{2}.kernelsize(2))';
net.layers{2}.b = num2cell(zeros(1,net.layers{2}.outputmaps));
%% Convolution-Subsampling layer
inputmaps = net.layers{2}.outputmaps;
fan_out = net.layers{3}.outputmaps * net.layers{3}.kernelsize(1);
fan_in = inputmaps * net.layers{3}.kernelsize(1);
csmaps = net.layers{3}.outputmaps/inputmaps;
for i = 1 : inputmaps 
    rng('shuffle');
    net.layers{3}.k{i} = mat2cell((rand(max(net.layers{3}.kernelsize),csmaps) - 0.5)...
        * 2 * sqrt(6 / (fan_in + fan_out)),net.layers{3}.kernelsize(1),ones(1,csmaps));
    net.layers{3}.b{i} = num2cell(zeros(1,csmaps));
end
%% Hidden layer
inputmaps = net.layers{3}.outputmaps;
fan_in = inputmaps*net.layers{3}.perNeural;%50*6
rng('shuffle');
net.layers{4}.k = mat2cell((rand(net.layers{4}.perNeural,fan_in)-0.5) * 2 ...
    * sqrt(6 / (100 + 300)),ones(1,net.layers{4}.perNeural),fan_in)';
net.layers{4}.b = zeros(net.layers{4}.outputmaps*net.layers{4}.perNeural,1); %change
%% Output layer
fvnum = net.layers{4}.perNeural;    %100
onum = 2;              % number of labels
rng('shuffle')
net.ffb = zeros(onum, 1);%change
rng('shuffle')
net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));%change
end
