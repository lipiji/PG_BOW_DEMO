clc; clear;

ini;

load('image_names', 'image_names');
load('labels', 'labels');
load('testset', 'testset');
load('trainset', 'trainset');
load('classes', 'classes');

num_class = length(classes);
num_imgs = length(labels);
for i = 1 : num_class
    class_name = classes{i,1};
    picstr = dir([pre_data_path, '/testing/', class_name, '/*.jpg']);
    
    [row,col] = size(picstr);
    picgather = cell(row,1);
    
    for j = 1 : row
        num_imgs = num_imgs + 1
        image_names{num_imgs}=['testing/', class_name,'/' picstr(j).name];
        labels(num_imgs,1)=i;
        trainset(num_imgs,1)=0;
        testset(num_imgs,1)=1;
    end
end

save('image_names','image_names');
save('labels','labels');
trainset=logical(trainset);
testset=logical(testset);
save('trainset','trainset');
save('testset','testset');