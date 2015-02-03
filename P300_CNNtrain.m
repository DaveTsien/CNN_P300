function net = P300_CNNtrain(net, x, y, opts)
for i = 1 : opts.trainepochs
    net = P300_CNNff1(net, reshape(x(:,:,i),64,78));
    net = P300_CNN_Dave_0(net, y(i,:));     %change
    net = P300_CNNapplygrads(net);
    if isempty(net.rL)
        net.rL(1) = net.L;
    end
    net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
    if isempty(net.rLs)
        net.rLs(1) = net.L;
    else
        net.rLs(end + 1) = net.L;
    end
end
end