function [y, a1, a2, a3, w2, w3, b2, b3, z1, z2, z3] = feedforward(lastlabel, x, w2, w3, b2, b3, z1, z2, z3)
%===========
n_neur3 = length(b3);
y = zeros(1, n_neur3);
y(lastlabel+1) = 1;

% a1 = double(rot90(x(:)))./256;
% feedforward
% layer 2
% for i=1:1:n_neur2
%     z2(i) = sum(w2(i,:) .* a1) + b2(i);
% end;
% a2 = 1./(1+exp(-z2));
% for i=1:1:n_neur3
%     z3(i) = w3(i,:) * a2(:) + b3(i);
% end;
% a3 = 1./(1+exp(-z3));

a1 = x;
z2 = transp(w2 * transp(a1)) + b2;
a2 = 1./(1+exp(-z2));
z3 = transp(w3 * transp(a2)) + b3;
a3 = 1./(1+exp(-z3));
