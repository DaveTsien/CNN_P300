function [opts, bad] = P300_CNNtest(net, x, y, opts)
    %  feedforward
    for i = 1 : opts.testepochs
        net = P300_CNNff1(net, reshape(x(:,:,i),64,78));
        opts.output(i,:) = net.o;
        if net.o(1) == net.o(2)
            bad = 1;
            opts.erp = [opts.erp -1];
        else            
            [~, h] = max(net.o);
            [~, a] = max(y(i,:));
            bad = find(h ~= a); 
            if bad == 1
                opts.erp = [opts.erp i];
            end
        end
        
        opts.er = opts.er + numel(bad);
    end
    opts.er = opts.er / opts.testepochs;
end