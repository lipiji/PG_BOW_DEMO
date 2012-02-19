function []=do_assignment(opts,assignment_opts)

display('Computing assignments');
assign_flag=1;
%% check if assignment already exists
try
    assignment_opts2=getfield(load([opts.globaldatapath,'/',assignment_opts.name,'_settings']),'assignment_opts');
    if(isequal(assignment_opts,assignment_opts2))
        assign_flag=0;
        display('Recomputing assignments for this settings');
    else
        display('Overwriting assignment with same name, but other Assignment settings !!!!!!!!!!');
    end
end

if(assign_flag)
    %% load data set information and vocabulary
    load(opts.image_names);
    nimages=opts.nimages;
    vocabulary=getfield(load([opts.globaldatapath,'/',assignment_opts.dictionary_type]),'dictionary');
    vocabulary_size=size(vocabulary,1);
    featuretype=assignment_opts.featuretype;
    
    %% apply assignment method to data set
    BOW=[];
    for ii=1:nimages
        image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(ii,3));                    % location where detector is saved
        inFName = fullfile(image_dir, sprintf('%s', featuretype));
        load(inFName, 'features');
        points = features.data;
        
        
        texton_ind.x = features.x;
        texton_ind.y = features.y;
        texton_ind.wid = features.wid;
        texton_ind.hgt = features.hgt;
        
        
        
        switch assignment_opts.type                                                         % select assignment method
            case '1nn'
                d2 = EuclideanDistance(points, vocabulary);
                [minz, index] = min(d2', [], 1);
                
                BOW(:,ii)=hist(index,(1:vocabulary_size));
                texton_ind.data = index;
                save ([image_dir,'/',assignment_opts.texton_name],'texton_ind');
                
            otherwise
                display('A non existing assignment method is selected !!!!!');
        end
        fprintf('Assign the %d th image\n',ii);
    end
    
    BOW=do_normalize(BOW,1);                                                                      % normalize the BOW histograms to sum-up to one.
    save ([opts.globaldatapath,'/',assignment_opts.name],'BOW');                               % save the BOW representation in opts.globaldatapath
    save ([opts.globaldatapath,'/',assignment_opts.name,'_settings'],'assignment_opts');
end
end