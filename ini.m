% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (pagelee.sd@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011

clear pg_opts
rootpath='E:\workspace\matlab\work\PG_BOW_DEMO\';

%%
addpath libsvm;
addpath BOW;
addpath AdaBoost;

%% change these paths to point to the image, data and label location
images_set=strcat(rootpath,'images');
data=strcat(rootpath,'data');
labels=strcat(rootpath,'labels');

%%
pg_opts.imgpath=images_set; % image path
pg_opts.datapath=data;
pg_opts.labelspath=labels;

%%
% local and global data paths
pg_opts.localdatapath=sprintf('%s/local',pg_opts.datapath);
pg_opts.globaldatapath=sprintf('%s/global',pg_opts.datapath);

% initialize the training set
pg_opts.trainset=sprintf('%s/trainset.mat',pg_opts.labelspath);
% initialize the test set
pg_opts.testset=sprintf('%s/testset.mat',pg_opts.labelspath);
% initialize the labels
pg_opts.labels=sprintf('%s/labels.mat',pg_opts.labelspath);
% initialize the image names
pg_opts.image_names=sprintf('%s/image_names.mat',pg_opts.labelspath);

% Classes
pg_opts.classes={...
    'Phoning'
    'PlayingGuitar'
    'RidingBike'
    'RidingHorse'
    'Running'
    'Shooting'
    };

pg_opts.nclasses=length(pg_opts.classes);

load(sprintf('%s',pg_opts.labels));
pg_opts.nimages=size(labels,1);

load(pg_opts.trainset);
load(pg_opts.testset);
pg_opts.ntraning =  length(find(trainset==1));
pg_opts.ntesting =  length(find(testset==1));

%% creat the directory to save data 
MakeDataDirectory(pg_opts);
