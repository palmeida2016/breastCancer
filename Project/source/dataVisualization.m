clear; clc; close all;

data = readtable('log.csv');
data = table2array(data);

% Plot Validation Accuracy
figure(1);
plot(data(:,1), data(:,2), data(:,1), data(:,5));
legend('Training','Validation','Location','best');
title('Accuracy');
xlabel('Epoch');
ylabel('Accuracy');


% Plot Validation Loss
figure(2);
plot(data(:,1), data(:,3), data(:,1), data(:,6));
legend('Training','Validation','Location','east');
title('Loss');
xlabel('Epoch');
ylabel('Loss');