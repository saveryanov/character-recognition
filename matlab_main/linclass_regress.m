close all;
clear all;

% Тренировочная выборка
M = dlmread('train.csv',';',1,0);
n_params = length(M(1,:));
Y = M(:, n_params);
n = length(Y);
M_changing = [M(:,1),M(:,3:4),M(:,6),M(:,8:12)];

I = [ones(n,1), M(:, 1:n_params-1), sin(2*pi*M_changing), cos(2*pi*M_changing), exp(M_changing),exp(-M(:,8))];
n_params = length(I(1,:));
n_params_static = n_params;

% /Нормировка
for i_power = 1:10:100
   
    if i_power>1
        I = [I, M_changing.^i_power];   
        n_params = length(I(1,:));
    end
    
    % Нормировка
    for p = 2:n_params 
        I(:,p) = I(:,p) / max(abs(I(:,p)))*3;
    end;

    % Инициализация параметров системы
    n_traintest = 1000; % кол-во примеров из тренировочной выборки для теста
    n_epochs = 10;
    error_epoch_test = zeros(1, n_epochs);
    error_epoch_train = zeros(1, n_epochs);
    theta = rand(1, n_params);
    nu = zeros(1, n_epochs);
    nu(1) = 81;
    for i = 2:1:n_epochs
        nu(i) = sqrt(nu(i-1));
    end;
    
    % Обучение
    for i_epoch = 1:1:n_epochs
        error_train_avrg = 0;
        error_test_avrg = 0;
        % обучение
        for t = 1:1:n-n_traintest
            result = round(sum(I(t,:).*theta)); % Вычисление выхода
            error_train_avrg = error_train_avrg + ((result - Y(t))^2)/(n-n_traintest)/2; % Подсчет ошибки
            % Обучение
            for k = 1:1:n_params
                theta(k) = theta(k) - 1/(n - n_traintest) * nu(i_epoch) * I(t,k) * (sum(I(t,:).*theta) - Y(t));
            end;
            %display(num2str([i_power, i_epoch, nu(i_epoch), t, result - Y(t), result, Y(t)]));
        end;
        

        % тестирование
        for t = n-n_traintest+1:1:n
            result = sum(I(t,:).*theta); % Вычисление выхода
            error_test_avrg = error_test_avrg + ((result - Y(t))^2)/n_traintest/2; % Подсчет ошибки\
        end;
        error_epoch_test(i_epoch) = error_test_avrg;
        error_epoch_train(i_epoch) = error_train_avrg;

        display(num2str([i_power, i_epoch, error_test_avrg]));

    end;
    % /Обучение

    % Тестирование
    % /Тестирование

    figure(i_power);
    hold on;
    grid on;
    plot(1:n_epochs, error_epoch_train);
    plot(1:n_epochs, error_epoch_test);
    legend('train','test');
end;