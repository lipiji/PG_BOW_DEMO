function [ ] = CalculateDictionary(opts, dictionary_opts)

fprintf('Building Dictionary using Training Data\n\n');

%% parameters
dictionary_flag=1;
dictionarySize = dictionary_opts.dictionarySize;
featureName=dictionary_opts.name;
featuretype=dictionary_opts.type;

try
    dictionary_opts2=getfield(load([opts.globaldatapath,'/',dictionary_opts.name,'_settings']),'dictionary_opts');
    if(isequal(dictionary_opts,dictionary_opts2))
        dictionary_flag=0;
        display(' dictionary has already been computed for this settings');
    else
        display('Overwriting  dictionary with same name, but other  dictionary settings !!!!!!!!!!');
    end
end


if(dictionary_flag)
    %% k-means clustering
    
    nimages=opts.ntraning;           % number of traning images in data set, we must make sure the fist nimages is for trarning
    
    niters=100;                     %maximum iterations
    
    image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(1,3)); % location descriptor
    inFName = fullfile(image_dir, sprintf('%s', featureName));
    load(inFName, 'features');
    data = features.data;
    
    image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(2,3)); % location descriptor
    inFName = fullfile(image_dir, sprintf('%s', 'sift_features'));
    load(inFName, 'features');
    data = [data;features.data];
    
    centres = zeros(dictionarySize, size(data,2));
    [ndata, data_dim] = size(data);
    [ncentres, dim] = size(centres);
    
    %% initialization
    
    perm = randperm(ndata);
    perm = perm(1:ncentres);
    centres = data(perm, :);
    
    num_points=zeros(1,dictionarySize);
    old_centres = centres;
    display('Run k-means');
    
    for n=1:niters
        % Save old centres to check for termination
        e2=max(max(abs(centres - old_centres)));
        
        inError(n)=e2;
        old_centres = centres;
        tempc = zeros(ncentres, dim);
        num_points=zeros(1,ncentres);
        
        for f = 1:nimages
            fprintf('The %d th interation the %d th image. eCenter=%f \n',n,f,e2);
            image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(f,3)); % location descriptor
            inFName = fullfile(image_dir, sprintf('%s',featureName));
            
            
            load(inFName, 'features');
            data = features.data;
            [ndata, data_dim] = size(data);
            
            id = eye(ncentres);
            d2 = EuclideanDistance(data,centres);
            % Assign each point to nearest centre
            [minvals, index] = min(d2', [], 1);
            post = id(index,:); % matrix, if word i is in cluster j, post(i,j)=1, else 0;
            
            num_points = num_points + sum(post, 1);
            
            for j = 1:ncentres
                tempc(j,:) =  tempc(j,:)+sum(data(find(post(:,j)),:), 1);
            end
            
        end
        
        for j = 1:ncentres
            if num_points(j)>0
                centres(j,:) =  tempc(j,:)/num_points(j);
            end
        end
        if n > 1
            % Test for termination
            
            %Threshold
            ThrError=0.009;
            
            if max(max(abs(centres - old_centres))) <0.009
                dictionary= centres;
                fprintf('Saving texton dictionary\n');
                save ([opts.globaldatapath,'/',featuretype],'dictionary');      % save the settings of descriptor in opts.globaldatapath
                break;
            end
            
            fprintf('The %d th interation finished \n',n);
        end
        
    end
    
    save ([opts.globaldatapath,'/',dictionary_opts.name,'_settings'],'dictionary_opts');
    
end
end
