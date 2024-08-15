% Load and preprocess data
data = readmatrix('DataSet-Full.xlsx');
x = data(:, 1:8); 
y = data(:, 9);
x2 = (x - min(x)) ./ (max(x) - min(x)); % Normalize input
y2 = log(1 + y); % Transform output
xt = x2';
yt = y2';

% Calculate variance of the original target variable
variance_y = var(y);

% Function to create and train the ANN without GUI
function [net, tr] = trainANN(xt, yt, hiddenLayerSizes, epochs, activationFunctions, lambda)
    net = patternnet(hiddenLayerSizes);
    
    % Set activation functions
    for i = 1:length(hiddenLayerSizes)
        net.layers{i}.transferFcn = activationFunctions{i};
    end
    net.layers{end}.transferFcn = 'purelin'; % Linear transfer function for the output layer
    
    net.trainFcn = 'trainbr'; % Bayesian Regularization backpropagation
    net.trainParam.showWindow = false; % Disable GUI
    net.trainParam.epochs = epochs; % Number of epochs
    net.performParam.regularization = lambda; % Set regularization parameter
    
    % Data division for training, validation, and testing
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    [net, tr] = train(net, xt, yt);
end

% Function to evaluate the ANN
function [rmse_train, rmse_test, mse_train, mse_test, normalized_mse_train, normalized_mse_test, yTrain, yTrainTrue, yTest, yTestTrue] = evaluateANN(net, tr, xt, yt, variance_y)
    yTrain = exp(net(xt(:, tr.trainInd))) - 1;
    yTrainTrue = exp(yt(tr.trainInd)) - 1;
    yTest = exp(net(xt(:, tr.testInd))) - 1;
    yTestTrue = exp(yt(tr.testInd)) - 1;
    rmse_train = sqrt(mean((yTrain - yTrainTrue).^2));
    rmse_test = sqrt(mean((yTest - yTestTrue).^2));
    mse_train = mean((yTrain - yTrainTrue).^2);
    mse_test = mean((yTest - yTestTrue).^2);
    normalized_mse_train = mse_train / variance_y;
    normalized_mse_test = mse_test / variance_y;
end

% Function to plot results
function plotResults(yTrainTrue, yTrain, yTestTrue, yTest, tr, net, xt, yt)
    figure;
    plot(yTrainTrue, yTrain, 'x'); hold on;
    plot(yTestTrue, yTest, 'o');
    plot(0:100, 0:100); hold off;
    xlabel('True Values');
    ylabel('Predicted Values');
    title('ANN Model Predictions');
    legend('Training Data', 'Testing Data', 'Ideal Fit', 'Location', 'southeast');
    grid on;

    figure;
    plotperform(tr);
    title('Training Performance');

    figure;
    plottrainstate(tr);
    title('Training State');

    figure;
    plotregression(yt(tr.trainInd), net(xt(:, tr.trainInd)), 'Training', ...
                   yt(tr.testInd), net(xt(:, tr.testInd)), 'Testing');
    title('Regression Analysis');

    figure;
    ploterrhist(yt - net(xt));
    title('Error Histogram');
end

% Define objective function for GA optimization
function rmse_test = objectiveFunction(params, xt, yt, activationFunctions)
    hiddenLayerSizes = round(params(1:3)); % First 3 values for hidden layer sizes
    lambda = params(4); % Last value for regularization parameter
    [net, tr] = trainANN(xt, yt, hiddenLayerSizes, 1, activationFunctions, lambda); % Train for 1 epoch per iteration
    yTest = exp(net(xt(:, tr.testInd))) - 1; % Predictions
    yTestTrue = exp(yt(tr.testInd)) - 1;
    rmse_test = sqrt(mean((yTest - yTestTrue).^2));
end

% GA options
options = optimoptions('ga', ...
                       'Display', 'iter', ...
                       'MaxGenerations', 50, ...
                       'MaxStallGenerations', 10, ...
                       'FunctionTolerance', 1e-4, ...
                       'UseParallel', false);

% Activation functions to test
activationFunctions = {'tansig', 'logsig', 'logsig'};
hiddenLayerSizeBounds = [1, 10; 1, 20; 1, 20]; % Different bounds for multiple layers
lambdaBounds = [0, 1]; % Regularization parameter bounds

% Run GA to optimize hidden layer sizes and regularization parameter
nvars = length(hiddenLayerSizeBounds) + 1;
lb = [hiddenLayerSizeBounds(:, 1); lambdaBounds(1)];
ub = [hiddenLayerSizeBounds(:, 2); lambdaBounds(2)];
[optimalParams, ~] = ga(@(x) objectiveFunction(x, xt, yt, activationFunctions), nvars, [], [], [], [], lb, ub, [], options);
optimalNeurons = round(optimalParams(1:3));
optimalLambda = optimalParams(4);

% Train final model with optimal hidden layer sizes and regularization parameter
[net, tr] = trainANN(xt, yt, optimalNeurons, 100, activationFunctions, optimalLambda); % Train for 100 epochs

% Evaluate the ANN
[rmse_train, rmse_test, mse_train, mse_test, normalized_mse_train, normalized_mse_test, yTrain, yTrainTrue, yTest, yTestTrue] = evaluateANN(net, tr, xt, yt, variance_y);

% Display results
disp(['Optimal hidden layer neurons: ', num2str(optimalNeurons)]);
disp(['Optimal regularization parameter: ', num2str(optimalLambda)]);
disp(['Training RMSE: ', num2str(rmse_train)]);
disp(['Testing RMSE: ', num2str(rmse_test)]);
disp(['Training MSE: ', num2str(mse_train)]);
disp(['Testing MSE: ', num2str(mse_test)]);
disp(['Normalized Training MSE: ', num2str(normalized_mse_train)]);
disp(['Normalized Testing MSE: ', num2str(normalized_mse_test)]);

% Plotting and visualization
plotResults(yTrainTrue, yTrain, yTestTrue, yTest, tr, net, xt, yt);