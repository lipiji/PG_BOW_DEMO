% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (peegeelee@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% classification script using SVM

fprintf('\nClassification using Pyramid BOW rbf_svm\n');
% load the BOW representations, the labels, and the train and test set
load(pg_opts.trainset);
load(pg_opts.testset);
load(pg_opts.labels);


%% sift
load([pg_opts.globaldatapath,'/',pyramid_opts.name])
train_labels    = labels(trainset);          % contains the labels of the trainset
train_data      = pyramid_all(:,trainset)';          % contains the train data
[train_labels,sindex]=sort(train_labels);    % we sort the labels to ensure that the first label is '1', the second '2' etc
train_data=train_data(sindex,:);
test_labels     = labels(testset);           % contains the labels of the testset
test_data       = pyramid_all(:,testset)';           % contains the test data


%% here you should of course use crossvalidation !

%%
bestcv = 0;
bestc=200;bestg=2;

options=sprintf('-s 0 -t 2 -c %f -b 1 -g %f -q',bestc,bestg);
model=svmtrain(train_labels,train_data,options);


[predict_label, accuracy , dec_values] = svmpredict(test_labels,test_data, model,'-b 1');
