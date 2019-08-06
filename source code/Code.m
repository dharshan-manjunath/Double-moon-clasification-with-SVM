%Author: Manjunath Dharshan
%Code Implementation for SVM with non linear data points
clear all; close all;
radius=13; % Outer Radius = r + (w)/2 Inner Radius = r - (w)/2
width=6; % Width
N=1000; % Number of Training Samples
TrainSample1=10*N;
distance=-12;% distance between two half moon
rng(20);
done=0; 
tmp1=[];
TrainSample=N;
while ~done, 
    tmp=[2*(radius+width/2)*(rand(TrainSample1,1)-0.5) (radius+width/2)*rand(TrainSample1,1)];
    tmp(:,3)=sqrt(tmp(:,1).*tmp(:,1)+tmp(:,2).*tmp(:,2)); 
    idx=find([tmp(:,3)>radius-width/2] & [tmp(:,3)<radius+width/2]);
    tmp1=[tmp1;tmp(idx,1:2)];
    if length(idx)>= TrainSample, 
        done=1;
    end
end
dataTrain=[tmp1(1:TrainSample,:) ones(TrainSample,1)*-1;[tmp1(1:TrainSample,1)+radius -tmp1(1:TrainSample,2)-distance ones(TrainSample,1)]];
R=radius+width/2;

%plotting the double moon
figure(1);
plot(dataTrain(dataTrain(:,3)==1,1),dataTrain(dataTrain(:,3)==1,2),'g*'); hold on;
plot(dataTrain(dataTrain(:,3)==-1,1),dataTrain(dataTrain(:,3)==-1,2),'r+'); 
set(gca, 'DataAspectRatio', [1,1,1]);
title(['Train Sample Double Moon with Inner Radius = 10, Outer radius=' num2str(R) ' width=' num2str(width) ' distance=' num2str(distance) ' Train Sample=' num2str(TrainSample)]);

%Creating Training matrix
X_train=vertcat(dataTrain(1:600,1:2),dataTrain(1001:1600,1:2));
Y_train=vertcat(dataTrain(1:600,3),dataTrain(1001:1600,3));
X_test=vertcat(dataTrain(601:1000,1:2),dataTrain(1601:2000,1:2));
Y_test=vertcat(dataTrain(601:1000,3),dataTrain(1601:2000,3));
rng(10);

% 10-fold Cross Validation
c = cvpartition(1200,'KFold',10);

%Train the SVM Classifier for 3 different Kernal functions
% Optimisation parameters Initilization
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
% SVM Model 1 with Radial Basis Kernal
svmmod1 = fitcsvm(X_train,Y_train,'KernelFunction','rbf','BoxConstraint',1,...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
% SVM Model 2 with tan hyperbolic function
svmmod2 = fitcsvm(X_train,Y_train,'KernelFunction','mysigmoid','BoxConstraint',0.01);
% SVM Model 3 with GHI_Kernel
svmmod3 = fitcsvm(X_train,Y_train,'KernelFunction','GHI_Kernel','BoxConstraint',0.01);
% Loss in cross validation calculation for RBF Kernel
lossnew = kfoldLoss(fitcsvm(X_train,Y_train,'CVPartition',c,'KernelFunction','rbf',...
    'BoxConstraint',svmmod1.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod1.HyperparameterOptimizationResults.XAtMinObjective.KernelScale));

% Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X_train(:,1)):d:max(X_train(:,1)),...
    min(X_train(:,2)):d:max(X_train(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores1] = predict(svmmod1,xGrid);
[~,scores2] = predict(svmmod2,xGrid);
[~,scores3] = predict(svmmod3,xGrid);

%Predictiting the labels with Test Data
y_pred1 = predict(svmmod1,X_test);
y_pred2 = predict(svmmod2,X_test);
y_pred3 = predict(svmmod3,X_test);

%Plots for Train samples decesion boundary and support vectors along with
%classification on test data
figure;
subplot(3,2,1);
h = nan(3,1); % Preallocation
h(1:2) = gscatter(X_train(:,1),X_train(:,2),Y_train,'rg','+*');
hold on
h(3) = plot(X_train(svmmod1.IsSupportVector,1),...
    X_train(svmmod1.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k');
%imagesc([min(X_train(:,1)) max(X_train(:,1))], [min(X_train(:,2)) max(X_train(:,2))], Ynp);
legend(h,{'-1','+1','Support Vectors'},'Location','Southeast');
title('Trained SVM Model With RBF Kernel C1=0.01');
hold on;
subplot(3,2,3);
g = nan(3,1); % Preallocation
g(1:2) = gscatter(X_train(:,1),X_train(:,2),Y_train,'rg','+*');
hold on
g(3) = plot(X_train(svmmod2.IsSupportVector,1),...
    X_train(svmmod2.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores2(:,2),size(x1Grid)),[0 0],'k');
legend(g,{'-1','+1','Support Vectors'},'Location','Southeast');
title('Trained SVM Model With TanH Kernel C1=0.01');
hold on;
subplot(3,2,5);
i = nan(3,1); % Preallocation
i(1:2) = gscatter(X_train(:,1),X_train(:,2),Y_train,'rg','+*');
hold on;
i(3) = plot(X_train(svmmod3.IsSupportVector,1),...
    X_train(svmmod3.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores3(:,2),size(x1Grid)),[0 0],'k');
legend(i,{'-1','+1','Support Vectors'},'Location','Southeast');
title('Trained SVM Model With GHI Kernel C1=0.01');
hold on;
subplot(3,2,2);
h = nan(3,1); % Preallocation
h(1:2) = gscatter(X_test(:,1),X_test(:,2),y_pred1,'rg','+*');
mis = ~(y_pred1==Y_test);
title('Test SVM Model With RBF Kernel C1=0.01');
hold on;
subplot(3,2,4);
h = nan(3,1); % Preallocation
h(1:2) = gscatter(X_test(:,1),X_test(:,2),y_pred2,'rg','+*');
mis = ~(y_pred2==Y_test);
title('Test SVM Model With TanH Kernel C1=0.01');
hold on;
subplot(3,2,6);
h = nan(3,1); % Preallocation
h(1:2) = gscatter(X_test(:,1),X_test(:,2),y_pred3,'rg','+*');
mis = ~(y_pred3==Y_test);
title('Test SVM Model With GHI Kernel C1=0.01');
hold off

Y_test(Y_test==-1) = 0;
y_pred2(y_pred2==-1) = 0;
y_pred2(y_pred2==-1) = 0;
y_pred2(y_pred2==-1) = 0;

% Confusion Matix generation
cm1 = confusionmat(Y_test,y_pred1);
cm2 = confusionmat(Y_test,y_pred2);
cm3 = confusionmat(Y_test,y_pred3);
%Plots for Confusion Matrix
figure;
plotconfusion(Y_test',y_pred1');
title('Confusion matrix for SVM Model with kernel rbf C1=0.01');
figure;
plotconfusion(Y_test',y_pred2');
title('Confusion matrix for SVM Model with kernel TanH C1=0.01');
figure;
plotconfusion(Y_test',y_pred3');
title('Confusion matrix for SVM Model with GHI kernel C1=0.01');






