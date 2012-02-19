function I = sp_load_image(image_fname)

I = imread(image_fname);
if ndims(I) == 3
    I = im2double(rgb2gray(I));
else
    I = im2double(I);
end
