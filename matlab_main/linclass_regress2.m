close all;
clear all;




% Тренировочная выборка
M = dlmread('train.csv',';',1,0);
figure (1)
hold on
% for j = 1:12
%     %figure(j)
%     %hold on
%     un_1 = unique (M(:,j));
%     for i=1:length(un_1)
%         stem(un_1(i),sum(M(:,13).*(M(:,j)==un_1(i)))/sum(M(:,j)==un_1(i)))
%     end
%     %hold off
% %     figure(j+100)
% %     stem(M(:,j),M(:,13));
% end

n_params = length(M(1,:));
Y = M(:, n_params);
n = length(Y);
M_changing = [M(:,1),M(:,3:4),M(:,9:12)];

I = [ones(n,1), M(:, 1:n_params-1), exp(M(:, 1:n_params-1)), sin(2*pi*M(:, 1:n_params-1)) ];
n_params = length(I(1,:));
n_params_static = n_params;

n_epochs = 10;
nu = zeros(1, n_epochs);
nu(1) = 100;
for i = 2:1:n_epochs
    nu(i) = (nu(i-1))/i;
end;
n_traintest = 1000; % кол-во примеров из тренировочной выборки для теста


for i_power = 1:1:10
   
    if i_power>1
        I = [I, M_changing.^i_power];   
        n_params = length(I(1,:));
    end
    
    % Нормировка
    for p = 2:n_params 
        I(:,p) = I(:,p) / max(abs(I(:,p)));
    end;
    

    % Инициализация параметров системы
   
    error_epoch_test = zeros(1, n_epochs);
    error_epoch_train = zeros(1, n_epochs);
    theta = 1*rand(1, n_params);
    
    % Обучение
    for i_epoch = 1:1:n_epochs
        error_train_avrg = 0;
        error_test_avrg = 0;
        % обучение
        for t = 1:1:n-n_traintest
            % Обучение
            theta  = theta -  1/(n - n_traintest) .* nu(i_epoch) .* I(t,:) .* (sum(I(t,:).*theta) - Y(t));
            %display(num2str([i_power, i_epoch, nu(i_epoch), t, result - Y(t), result, Y(t)]));
        end;
        % тестирование на тренировочной выборке
        for t = 1:n-n_traintest
            error_train_avrg = error_train_avrg + ((sum(I(t,:).*theta) - Y(t))^2)/(n-n_traintest)/2; % Подсчет ошибки\
        end;
        % тестирование на тестовом фрагменте
        for t = n-n_traintest+1:1:n
            error_test_avrg = error_test_avrg + ((sum(I(t,:).*theta) - Y(t))^2)/n_traintest/2; % Подсчет ошибки\
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