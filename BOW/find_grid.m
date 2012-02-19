function [grid_x, grid_y] = find_grid(hgt, wid, grid_spacing, patch_size)
%% make grid (coordinates of upper left patch corners)

rem_x = mod(wid-patch_size,grid_spacing);
offset_x = floor(rem_x/2)+1;
rem_y = mod(hgt-patch_size,grid_spacing);
offset_y = floor(rem_y/2)+1;

%% make grid (coordinates of upper left patch corners)
[grid_x,grid_y] = meshgrid(offset_x:grid_spacing:wid-patch_size+1, offset_y:grid_spacing:hgt-patch_size+1);
