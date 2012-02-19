clc; clear;
load image_names;
load labels;
load testset;
load trainset;

 %image_names=[];
 %labels=[];
 %testset=[];
 %trainset=[];

picstr=dir('E:\workspace\matlab\work\BOW_DEMO\images\training\Phoning\*.jpg');
[row,col]=size(picstr);
picgather=cell(row,1);

ci=1;


for i=(ci-1)*40+1:ci*40
    image_names{i}=['training\Phoning\',picstr(i-(ci-1)*40).name];
    labels(i,1)=ci;
    trainset(i,1)=1;
    testset(i,1)=0;
end


save('image_names','image_names');
save('labels','labels');
trainset=logical(trainset);
testset=logical(testset);
save('trainset','trainset');
save('testset','testset');


