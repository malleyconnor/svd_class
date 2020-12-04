im_size = [150, 150, 3];
im_shape = [150 150];

% # of 'training' images per class
num_per_class = 25;

% # of testing images per class
num_per_class_test = 10;

% # of top eigen images to project onto for prediction
rank = 1;


% Reading in all class samples
class_strs = ["mountain", "sea", "street", "buildings", "forest", "glacier"];
class_tensors = make_class_tensors('.\intel_image\seg_train\seg_train', class_strs, num_per_class, im_shape);
for j=1:length(class_tensors)
    avg_images{j}   = avg_image(class_tensors{j});
    subplot(3,2, j)
    imshow(uint8(avg_images{j}), 'InitialMagnification', 1000);
    title(class_strs{j});
end


% Getting SVDs of each class
svd_matrices=get_svd_matrices(class_tensors, avg_images);
for j=1:length(svd_matrices)
    svds{j} = {}; svds{j}{1}={}; svds{j}{2}={}; svds{j}{3}={};
    
    [svds{j}{1}{1}, svds{j}{1}{2}, svds{j}{1}{3}] = svd(svd_matrices{j}{1}, 'econ');
    [svds{j}{2}{1}, svds{j}{2}{2}, svds{j}{2}{3}] = svd(svd_matrices{j}{1}, 'econ');
    [svds{j}{3}{1}, svds{j}{3}{2}, svds{j}{3}{3}] = svd(svd_matrices{j}{1}, 'econ');
end


% Plotting eigen images to defined rank
figure();
im_count = 1;
sgtitle('Eigen Images of Classes');
for j=1:length(svd_matrices)
    for k=1:rank
        subplot(length(svd_matrices),rank, im_count)
        this_r = reshape(svds{j}{1}{1}(:,k), im_size(1), im_size(2));
        this_g = reshape(svds{j}{2}{1}(:,k), im_size(1), im_size(2));
        this_b = reshape(svds{j}{3}{1}(:,k), im_size(1), im_size(2));
        this_eigen_img = uint8(cat(3, this_r, this_g, this_b).*im_size(1).*im_size(2) + avg_images{j});
        imshow(this_eigen_img);
        if k == 1
            ylabel(class_strs{j});
            set(get(gca,'ylabel'),'rotation',0);
        end
        if j == 1
            title(sprintf('Eigen Image #%d', k));
        end
        im_count = im_count + 1;
    end
end

% Getting predictions
[test_images, test_labels] = get_test_images('.\intel_image\seg_test\seg_test', ...
    class_strs, num_per_class_test, [150 150]);


predictions = zeros([1 length(test_images)]);
for j=1:length(test_images)
    predictions(j) = predict(test_images{j}, svds, avg_images, rank);
end

disp(predictions);
disp(test_labels);
prediction_acc = double((length(test_labels) - nnz(predictions - test_labels))) / length(test_labels);
disp(prediction_acc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reforms training set tensors into input matrices for SVD
function svd_matrices=get_svd_matrices(class_tensors, avg_images)
    for j=1:length(class_tensors)
        % Getting avg image of each class
        avg_images{j}   = avg_image(class_tensors{j});

        % Zero mean un-folded images (input matrices for SVD)
        zm_tensor = zeros(size(class_tensors{j}));
        for k=1:length(class_tensors{j}(1,1,1,:))
            zm_tensor(:,:,:,k) = class_tensors{j}(:,:,:,k) - avg_images{j};
        end

        % Unfolding the zmean tensor to get matrices for SVD
        matrix_size = size(zm_tensor(:,:,:,:)); 
        matrix_size(1) = matrix_size(1)*matrix_size(2); matrix_size(2) = matrix_size(4); 
        matrix_size = matrix_size(1:2);
        svd_matrices{j} = {zeros(matrix_size), zeros(matrix_size), zeros(matrix_size)};
        for k=1:length(class_tensors{j}(1,1,1,:))
            svd_matrices{j}{1}(:,k) = reshape(zm_tensor(:,:,1,k), length(zm_tensor(:,1,1,1))*length(zm_tensor(1,:,1,1)), 1);
            svd_matrices{j}{2}(:,k) = reshape(zm_tensor(:,:,2,k), length(zm_tensor(:,1,2,1))*length(zm_tensor(1,:,2,1)), 1);
            svd_matrices{j}{3}(:,k) = reshape(zm_tensor(:,:,3,k), length(zm_tensor(:,1,3,1))*length(zm_tensor(1,:,3,1)), 1);
        end
    end

end

function [test_images, test_labels]=get_test_images(test_dir, class_strs, num_per_class, im_shape)
    test_images={};
    test_labels=zeros([1 num_per_class*length(class_strs)]);
   
    img_ind=1;
    for class=1:length(class_strs)
        % Probably a way to shorten this bit
        % Gets list of filenames
        jpg_files = ls(test_dir+"\"+strtrim(class_strs(class))+"\*.jpg");
        file_strs = strings(length(jpg_files(:,1)), 1);
        for j=1:length(jpg_files(:,1))
            file_strs(j) = string(jpg_files(j,:));
        end
        %file_strs = file_strs(randperm(numel(file_strs)));
        
    
        j=1;
        for k=1:length(file_strs)
            im_in = imread(test_dir+"\"+strtrim(class_strs(class))+"\"+file_strs(k));
            
            % Ensures input images are desired shape (150x150x3 for intel_image set)
            % Skips if it is not
            if (~isequal(length(size(im_in)), length([im_shape 3])) || (~isequal(size(im_in), [im_shape 3])))
                continue
            end
            if length(im_in(:,1,1)) ~= 150 || length(im_in(1,:,1)) ~= 150
                continue
            end
            
            test_images{img_ind} = im_in;
            test_labels(img_ind) = class;
            img_ind = img_ind+1;
     
            % Breaks if we have enough samples
            if j == num_per_class
                break
            else
                j = j+1;
            end
        end % class samples loop
    end % class loop
end

% Projects onto bases of each class and gets prediction from RMSE
function prediction=predict(img, svds, avg_images, rank)
    prediction=0;
    max_score=0;
    scores = zeros(length(avg_images), 1);
    for class=1:length(svds)
        zm_img = double(img);% - avg_images{class};
        r = double(reshape(zm_img(:,:,1), length(img(:,1,1))*length(img(1,:,1)), 1));
        g = double(reshape(zm_img(:,:,2), length(img(:,1,2))*length(img(1,:,2)), 1));
        b = double(reshape(zm_img(:,:,3), length(img(:,1,3))*length(img(1,:,3)), 1));
    
        r_proj = r' * svds{class}{1}{1}(:, 1:rank);
        g_proj = g' * svds{class}{2}{1}(:, 1:rank);
        b_proj = b' * svds{class}{3}{1}(:, 1:rank);

        scores(class) = (mean(abs(r_proj)) + mean(abs(g_proj)) + mean(abs(b_proj)))/3;
    end
    [score, prediction] = max(scores);
end
