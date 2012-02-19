% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (peegeelee@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% classification script using SVM

fprintf('\nClassification using BOW rbf_svm\n');
% load the BOW representations, the labels, and the train and test set
load(pg_opts.trainset);
load(pg_opts.testset);
load(pg_opts.labels);
load([pg_opts.globaldatapath,'/',assignment_opts.name])


train_labels    = labels(trainset);          % contains the labels of the trainset
train_data      = BOW(:,trainset)';          % contains the train data
[train_labels,sindex]=sort(train_labels);    % we sort the labels to ensure that the first label is '1', the second '2' etc
train_data=train_data(sindex,:);
test_labels     = labels(testset);           % contains the labels of the testset
test_data       = BOW(:,testset)';           % contains the test data



%% here you should of course use crossvalidation !

%%
bestc=200;bestg=2;
bestcv=0;
% for log2c = -1:10,
%   for log2g = -1:0.1:1.5,
%     cmd = ['-v 5 -t 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = svmtrain(train_labels, train_data, cmd);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end


options=sprintf('-s 0 -t 2 -c %f -b 1 -g %f -q',bestc,bestg);
model=svmtrain(train_labels, train_data,options);


[predict_label, accuracy , dec_values] = svmpredict(test_labels,test_data, model,'-b 1');

