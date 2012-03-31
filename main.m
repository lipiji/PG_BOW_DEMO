%% Script to perform BOW-based image classification demo
% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (pagelee.sd@gmail.com)  
% Blog: иЇзг http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% initialize the settings
display('*********** start *********')
clc;
ini;
detect_opts=[];descriptor_opts=[];dictionary_opts=[];assignment_opts=[];ada_opts=[];


%% Descriptors
descriptor_opts.type='sift';                                                     % name descripto
descriptor_opts.name=['des',descriptor_opts.type]; % output name (combines detector and descrtiptor name)
descriptor_opts.patchSize=16;                                                   % normalized patch size
descriptor_opts.gridSpacing=8; 
descriptor_opts.maxImageSize=1000;
GenerateSiftDescriptors(pg_opts,descriptor_opts);

%% Create the texton dictionary
dictionary_opts.dictionarySize = 300;
dictionary_opts.name='sift_features';
dictionary_opts.type='sift_dictionary';
CalculateDictionary(pg_opts, dictionary_opts);

%% assignment
assignment_opts.type='1nn';                                 % name of assignment method
assignment_opts.descriptor_name=descriptor_opts.name;       % name of descriptor (input)
assignment_opts.dictionary_name=dictionary_opts.name;       % name of dictionary
assignment_opts.name=['BOW_',descriptor_opts.type];         % name of assignment output
assignment_opts.dictionary_type=dictionary_opts.type;
assignment_opts.featuretype=dictionary_opts.name;
assignment_opts.texton_name='texton_ind';
do_assignment(pg_opts,assignment_opts);

%% CompilePyramid
pyramid_opts.name='spatial_pyramid';
pyramid_opts.dictionarySize=dictionary_opts.dictionarySize;
pyramid_opts.pyramidLevels=3;
pyramid_opts.texton_name=assignment_opts.texton_name;
CompilePyramid(pg_opts,pyramid_opts);

%% Classification
do_classification_rbf_svm

%% histogram intersection kernel
do_classification_inter_svm
%% pyramid bow rbf
do_p_classification__rbf_svm   

%% pyramid bow histogram intersection kernel
do_p_classification__inter_svm
show_results_script

%% AdaBoost
ada_opts.T = 100;
ada_opts.weaklearner  = 0;
ada_opts.epsi = 0.2;
ada_opts.lambda = 1e-3;
ada_opts.max_ite = 3000;
ada_opts.bow = assignment_opts.name;
ada_opts.pbow = pyramid_opts.name;
do_classification_adaboost_bow(pg_opts,ada_opts);
%do_classification_adaboost_pyramid_bow(pg_opts,ada_opts);


