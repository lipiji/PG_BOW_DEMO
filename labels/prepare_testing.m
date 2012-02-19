clc; clear;
load image_names;
load labels;
load testset;
load trainset;

 %image_names=[];
 %labels=[];
 %testset=[];
 %trainset=[];

picstr=dir('E:\workspace\matlab\work\BOW_DEMO\images\testing\Phoning\*.jpg');
[row,col]=size(picstr);
picgather=cell(row,1);

training = 240;
ci=1;


for i=(ci-1)*20+1:ci*20
    image_names{training+i}=['testing\Phoning\',picstr(i-(ci-1)*20).name];
    labels(training+i,1)=ci;
    trainset(training+i,1)=0;
    testset(training+i,1)=1;
end


save('image_names','image_names');
save('labels','labels');
trainset=logical(trainset);
testset=logical(testset);
save('trainset','trainset');
save('testset','testset');


