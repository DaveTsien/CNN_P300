function net = P300_CNNapplygrads(net)
%% Convolution layer
net.layers{2}.k = mat2cell(cell2mat(net.layers{2}.k) + cell2mat(net.layers{2}.dk')',...
    1,ones(1,net.layers{2}.outputmaps)*net.layers{2}.kernelsize(2));
net.layers{2}.b = num2cell(cell2mat(net.layers{2}.b) + cell2mat(net.layers{2}.db));
%% Convolution-Subsampling layer
for i = 1 : numel(net.layers{2}.a)
    net.layers{3}.k{1,i} = mat2cell(cell2mat(net.layers{3}.k{1,i}) + cell2mat(net.layers{3}.dk{1,i}),...
        net.layers{3}.kernelsize(1),ones(1,5));
    net.layers{3}.b{1,i} = num2cell(cell2mat(net.layers{3}.b{1,i}) + cell2mat(net.layers{3}.db{1,i}));
end
%% hidden layer
net.layers{4}.k = mat2cell(cell2mat(net.layers{4}.k) + cell2mat(net.layers{4}.dk),...
    1,ones(1,net.layers{4}.perNeural)*net.layers{3}.perNeural*net.layers{3}.outputmaps);
net.layers{4}.b = net.layers{4}.b + net.layers{4}.db';
net.ffW = net.ffW + net.dffW;
net.ffb = net.ffb + net.dffb';
end
