function [error] = testNet(I,labels, w2, w3, b2, b3, z1, z2, z3)
%===========
error = 0;
for i_train = 1:1:length(I)
    x = I(i_train,:);
    lastlabel = labels(i_train);

    [~, ~, ~, a3, w2, w3, b2, b3, z1, z2, z3] = feedforward(lastlabel, x, w2, w3, b2, b3, z1, z2, z3);

    lastans = 0;
    for i=1:1:10
        if a3(i) == max(a3)
            lastans = i-1;
        end;
    end;
    
    error = error + ((lastans - lastlabel)^2)/length(I)/2;
    
end;

