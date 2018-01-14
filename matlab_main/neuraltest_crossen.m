close all;
clear all;

n_neur3 = 1000;

n_neur2 = 8;
n2_errors = zeros(1, n_neur2);
%for n_neur2 = 1:1:25

M = dlmread('train.csv',';',1,0);
n_params = length(M(1,:));
m = length(M(:,1));
n_last_tests = 1000;
labels = M(1:m-n_last_tests, n_params);

I = M(1:m-n_last_tests, 1:1:n_params);
I_test = M(m-n_last_tests+1:m, 1:1:n_params);
labels_test = M(m-n_last_tests+1:m, n_params);

n_neur1 = n_params;

z1 = ones(1, n_neur1);
z2 = ones(1, n_neur2);
z3 = ones(1, n_neur3);

a1 = ones(1, n_neur1);
a2 = ones(1, n_neur2);
a3 = ones(1, n_neur3);

b1 = ones(1, n_neur1);
b2 = ones(1, n_neur2);
b3 = ones(1, n_neur3);

w2 = rand(n_neur2, n_neur1)*2 - 1;
w3 = rand(n_neur3, n_neur2)*2 - 1;

nu_i = [1000, 1000, 500, 500, 100, 100, 10, 10, 10, 10, 10];
n_repeats = length(nu_i);

correct = zeros(1, n_last_tests);

epoch_error = zeros(1, n_repeats+1);
[epoch_error(1)] = testNet(I_test,labels_test, w2, w3, b2, b3, z1, z2, z3);
m = m - n_last_tests;
for i_repeat = 1:1:n_repeats
    nu = nu_i(i_repeat);
    for i_train = 1:1:m
        x = I(i_train,:);
        lastlabel = labels(i_train);

        [y, a1, a2, a3, w2, w3, b2, b3, z1, z2, z3] = feedforward(lastlabel, x, w2, w3, b2, b3, z1, z2, z3);

        lastans = find(a3 == max(a3)) - 1;

        if lastans == lastlabel
            correct(mod(i_train, n_last_tests)+1) = 1;
        else
            correct(mod(i_train, n_last_tests)+1) = 0;
        end;
        sum_corr = sum(correct);

        %display(num2str([i_train, i_repeat, epoch_error(i_repeat), sum_corr/n_last_tests, nu]));

        % calculate error
        error3 = a3 - y;

        % backpropagate error
        error2 = zeros(1, n_neur2);
        for i=1:1:n_neur2
            for j=1:1:n_neur3
              error2(i) = error2(i) + w3(j,i) * error3(j); 
            end;
        end;

        % correction
        b3 = b3 - nu / m * error3;

        for i=1:1:n_neur2
            for j=1:1:n_neur3
                w3(j, i) = w3(j, i) - nu / m * error3(j) * a2(i);
            end;
        end;

        b2 = b2 - nu / m * error2;


        for i=1:1:n_neur1        
            for j=1:1:n_neur2
                w2(j, i) = w2(j, i) - nu / m * error2(j) * a1(i);
            end;
        end;
    end;

    [epoch_error(i_repeat+1)] = testNet(I_test,labels_test, w2, w3, b2, b3, z1, z2, z3);
    display(num2str([i_train, epoch_error(i_repeat+1)]));

end;
%    n2_errors(n_neur2) = epoch_error(2);
%end;

% n_arg = 1:1:25;
% figure(1);
% plot(n_arg, 1-n2_errors);


% figure(2);
% plot(error_t);


%save('Asave.mat', 'test', '-mat');
%load Asave test
% 
% save('25a1.mat', 'a1', '-mat');
% save('25a2.mat', 'a2', '-mat');
% save('25a3.mat', 'a3', '-mat');
% 
% save('100b1.mat', 'b1', '-mat');
% save('100b2.mat', 'b2', '-mat');
% save('100b3.mat', 'b3', '-mat');
% 
% save('100w2.mat', 'w2', '-mat');
% save('100w3.mat', 'w3', '-mat');
% 
% save('epoch_error100.mat', 'epoch_error', '-mat');