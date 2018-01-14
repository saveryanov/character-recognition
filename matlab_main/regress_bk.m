close all;
clear all;

% Тренировочная выборка
M = dlmread('train.csv',';',1,0);
n_params = length(M(1,:));
Y = M(:, n_params);
n = length(Y);
n_params = n_params - 1;

%I = [ones(n,1), M(:, 1:1:n_params-1), (M(:, 1:1:n_params-1)).^5];
poweri = 12;
Ipow = [(M(:,2:1:2)).^poweri, M(:,4:1:5).^poweri, M(:,7).^poweri, M(:,9:1:n_params).^poweri];

cosi = 0;
cosn = 1.0;
cosb = 0.0;
Icos = [cos(M(:,2:1:2)*cosn + cosi)+cosb, cos(M(:,4:1:5)*cosn + cosi)+cosb, cos(M(:,7)*cosn + cosi)+cosb, cos(M(:,9:1:n_params)*cosn + cosi)+cosb];

sini = 0;
sinn = 1.0;
sinb = 0.0;
Isin = [sin(M(:,2:1:2)*sinn + sini)+sinb, sin(M(:,4:1:5)*sinn + sini)+sinb, sin(M(:,7)*sinn + sini)+sinb, sin(M(:,9:1:n_params)*sinn + sini)+sinb];


divi = 1;
Idiv = [divi./(M(:,2:1:2)+0.1), divi./(M(:,4:1:5)+0.1), divi./(M(:,7)+0.1), divi./(M(:,9:1:n_params)+0.1)];

I = [ones(n,1), M(:, 2:1:n_params), Ipow, Idiv];
%I = [ones(n,1), M(:, 2:1:n_params)];
n_params = length(I(1,:));

Inew3 = zeros(n,12);
for i = 1:1:n
    Inew3(i,I(i,3)) = 1;
end;
Inew = zeros(n,7);
for i = 1:1:n
    Inew(i,I(i,6)+1) = 1;
end;
Inew2 = zeros(n,4);
for i = 1:1:n
    Inew2(i,I(i,8)) = 1;
end;
Inew4 = zeros(n,24);
for i = 1:1:n
    Inew4(i,I(i,4)+1) = 1;
end;

I = [I(:,1:1:2), I(:,5), I(:,7), I(:,9:1:n_params), Inew, Inew2, Inew3, Inew4];

% Инициализация параметров системы
n_traintest = 1000; % кол-во примеров из тренировочной выборки для теста

nu = [ones(1, 40) * 14, ones(1, 20) * 5]
n_epochs = length(nu);
error_epoch_test = zeros(1, n_epochs);
error_epoch_train = zeros(1, n_epochs);
% Нормировка
minI = zeros(1,n_params);
maxI = zeros(1,n_params);
for p = 7:1:n_params 
    minI(p) = min(I(:,p));
    maxI(p) = max(I(:,p));
    I(:,p) = ((I(:,p) - min(I(:,p)))/(max(I(:,p)) - min(I(:,p))))*2 - 1;
    %I(:,p) = I(:,p)/max(I(:,p));
end;
%I = [I(:, 1:1:n_params), cos(I(:, 1:1:n_params))];

n_params = length(I(1,:));
theta = rand(1, n_params);

% Обучение
best_error_abs = 99999999;
best_theta = theta;
for i_epoch = 1:1:n_epochs
    error_train_avrg = 0;
    error_test_avrg = 0;
    error_test_abs = 0;
    % обучение
    batch_n = 5;
    for t = 1:batch_n:n-n_traintest
        % Обучение
        theta_batch = theta;
        for bt = 0:1:batch_n-1
            %display(num2str([t, bt, bt+t]));
            if bt+t > n-n_traintest
                break;
            end;
            theta_batch  = theta_batch -  1/(n - n_traintest) .* nu(i_epoch) .* I(t+bt,:) .* (sum(I(t+bt,:).*theta) - Y(t+bt));
        end;
        theta = theta_batch;
        %display(num2str([i_epoch, nu(i_epoch), t, result - Y(t), result, Y(t)]));
    end;
    % тестирование на тренировочной выборке
    for t = 1:n-n_traintest
        result = (sum(I(t,:).*theta));
        if result < 0 
            result = 0;
        end;
        error_train_avrg = error_train_avrg + ((result - Y(t))^2)/(n-n_traintest)/2; % Подсчет ошибки\
    end;
    % тестирование
    for t = n-n_traintest+1:1:n
        result = (sum(I(t,:).*theta));
        if result < 0 
            result = 0;
        end;
        error_test_abs = error_test_abs + abs(result - Y(t))/n_traintest/2; % Подсчет ошибки\
    end;
    % тестирование
    for t = n-n_traintest+1:1:n
        result = (sum(I(t,:).*theta));
        if result < 0 
            result = 0;
        end;
        error_test_avrg = error_test_avrg + ((result - Y(t))^2)/n_traintest/2; % Подсчет ошибки\
    end;
    error_epoch_test(i_epoch) = error_test_avrg;
    error_epoch_train(i_epoch) = error_train_avrg;
    if best_error_abs > error_test_abs
       best_error_abs = error_test_abs;
       best_theta = theta;
    end;    
    display(num2str([i_epoch, error_test_avrg, error_train_avrg, error_test_abs, best_error_abs]));

end;
% /Обучение

% Тестирование
% /Тестирование

figure(1);
hold on;
grid on;
plot(1:n_epochs, error_epoch_train);
plot(1:n_epochs, error_epoch_test);
legend('train','test');

res = zeros(3,n_traintest);
for t = n-n_traintest+1:1:n
    res(1,t - n+n_traintest) = Y(t);
    res(2,t - n+n_traintest) = round(sum(I(t,:).*theta));
    res(3,t - n+n_traintest) = round(sum(I(t,:).*theta))-Y(t);
end;