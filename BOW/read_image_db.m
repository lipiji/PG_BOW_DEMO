function [im]=read_image_db(opts,imIndex)
% reads images from data base indicated in opts

load(opts.image_names);

try
    im=imread([opts.imgpath,'/', image_names{imIndex}]);
    im=double(im);
catch
    display('image does not exist');
    im=[];
end