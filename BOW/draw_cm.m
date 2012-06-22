% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (peegeelee@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
function draw_cm(mat,tick,num_class)
%%
%  作者： 丕子  zhizhihu.com
%  参数：mat-矩阵；tick-要在坐标轴上显示的label向量，例如{'label_1','label_2'...}
%
%%
imagesc(1:num_class,1:num_class,mat);            %# 绘彩色图
colormap(flipud(gray));  %# 转成灰度图，因此高value是渐黑色的，低value是渐白的

textStrings = num2str(mat(:),'%0.2f');  
textStrings = strtrim(cellstr(textStrings)); 
[x,y] = meshgrid(1:num_class); 
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim')); 
textColors = repmat(mat(:) > midValue,1,3); 
%改变test的颜色，在黑cell里显示白色
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'xticklabel',tick,'XAxisLocation','top');
set(gca, 'XTick', 1:num_class, 'YTick', 1:num_class); % to handle a bug
set(gca,'yticklabel',tick);

%% rotate x label
rotateXLabels(gca, 315 );



