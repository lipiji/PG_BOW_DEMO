function []=MakeDataDirectory(opts)
% makes a directory in 'opts.datapath' descriptors containing 
% a directory for all images in the dataset

% /data
if exist([opts.datapath],'dir')~=7
    mkdir(opts.datapath)
end


%/data/global
if exist([opts.datapath,'/global'],'dir')~=7
    mkdir(opts.datapath,'global')
end

%/data/local
if exist([opts.datapath,'/local'],'dir')~=7 % if the dir is not exist, then create it
    mkdir(opts.datapath,'local')
    for ii=1:opts.nimages
        mkdir(sprintf('%s/local',opts.datapath),num2string(ii,3));     
    end
end



        