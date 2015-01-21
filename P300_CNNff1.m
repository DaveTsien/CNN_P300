function net = P300_CNNff1(net, x)
%% Convolution layer
net.layers{1}.a{1} = x; 
net.layers{2}.a = mat2cell(1.7159*tanh(2/3*(cell2mat(net.layers{2}.k')*net.layers{1}.a{1}...
    +repmat(cell2mat(net.layers{2}.b'),1,net.layers{2}.perNeural))),ones(1,net.layers{2}.outputmaps),net.layers{2}.perNeural)';
%% Convolution-Subsampling layer
inputmaps = net.layers{2}.outputmaps;
for i = 1 : inputmaps   %10
    w = reshape(net.layers{2}.a{i}', max(net.layers{3}.kernelsize), net.layers{3}.perNeural)';%6*13,reshape caution!
    net.layers{3}.a(5*(i-1)+1:5*i) = mat2cell(1.7159*tanh(2/3*(w*cell2mat(net.layers{3}.k{i})...
        +repmat(cell2mat(net.layers{3}.b{i}),net.layers{3}.perNeural,1)))',ones(1,5),net.layers{3}.perNeural);
end    
%% Hidden layer
csa = cell2mat(net.layers{3}.a);    % 1*300
hk = cell2mat(net.layers{4}.k');    % 100*300
z = csa*hk'; %1*100 
net.layers{4}.a = sigm(z + (net.layers{4}.b)');
%% Output layer
z = net.layers{4}.a*(net.ffW)';
net.o = sigm(z + net.ffb');
end