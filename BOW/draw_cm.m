function draw_cm(mat,tick,num_class)
%%
%
%%
imagesc(1:num_class,1:num_class,mat);
colormap(flipud(gray));
textStrings = num2str(mat(:),'%0.2f');  
textStrings = strtrim(cellstr(textStrings)); 
[x,y] = meshgrid(1:num_class); 
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim')); 
textColors = repmat(mat(:) > midValue,1,3); 
set(hStrings,{'Color'},num2cell(textColors,2));  % Change the text colors

set(gca,'xticklabel',tick,'XAxisLocation','top');
set(gca, 'XTick', 1:num_class, 'YTick', 1:num_class); % to handle a bug
set(gca,'yticklabel',tick);

%% rotate x label
rotateXLabels(gca, 315 );



