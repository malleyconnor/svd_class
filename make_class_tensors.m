% Creates 4D-Tensors containing stacked RGB images belonging to a unique class.
function class_tensors=make_class_tensors(train_dir, class_strs, num_per_class, im_shape)
    class_tensors={};
   
    for class=1:length(class_strs)
        % Probably a way to shorten this bit
        % Gets list of filenames
        jpg_files = ls(train_dir+"\"+strtrim(class_strs(class))+"\*.jpg");
        file_strs = strings(length(jpg_files(:,1)), 1);
        for j=1:length(jpg_files(:,1))
            file_strs(j) = string(jpg_files(j,:));
        end
        file_strs = file_strs(randperm(numel(file_strs)));
        
        % Reads files into tensors
        class_tensors{class} = zeros([im_shape 3 num_per_class], 'double');
        j=1;
        for k=1:length(file_strs)
            im_in = imread(train_dir+"\"+strtrim(class_strs(class))+"\"+file_strs(k));
            
            % Ensures input images are desired shape (150x150x3 for intel_image set)
            % Skips if it is not
            if (~isequal(length(size(im_in)), length([im_shape 3])) || (~isequal(size(im_in), [im_shape 3])))
                continue
            end
            if length(im_in(:,1,1)) ~= 150 || length(im_in(1,:,1)) ~= 150
                continue
            end
            
            class_tensors{class}(:,:,:,j) = im_in;
     
            % Breaks if we have enough samples
            if j == num_per_class
                break
            else
                j = j+1;
            end
        end % class samples loop
    end % class loop
end