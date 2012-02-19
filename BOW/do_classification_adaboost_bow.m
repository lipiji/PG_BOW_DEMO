function do_classification_adaboost_bow(pg_opts,options)
% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (peegeelee@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% classification script using SVM

fprintf('\nClassification using BOW AdaBoost\n');
% load the BOW representations, the labels, and the train and test set
load(pg_opts.trainset);
load(pg_opts.testset);
load(pg_opts.labels);
load([pg_opts.globaldatapath,'/',options.bow])


train_labels    = labels(trainset);          % contains the labels of the trainset
train_data      = BOW(:,trainset);          % contains the train data
[train_labels,sindex]=sort(train_labels);    % we sort the labels to ensure that the first label is '1', the second '2' etc
train_data=train_data(:,sindex);
test_labels     = labels(testset);           % contains the labels of the testset
test_data       = BOW(:,testset);           % contains the test data


model                   = gentleboost_model(train_data , train_labels' , options.T , options);
[yest , fx]             = gentleboost_predict(test_data, model , options);
num_pre = sum(test_labels' == (yest+1));
Perf                    =num_pre /length(test_labels);
fprintf('Accuracy = %f (%d/%d) (classification)\n',Perf*100,num_pre,length(test_labels));

end

