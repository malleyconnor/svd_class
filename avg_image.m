% Gets average image of tensor of RGB images. (shape = heightxwidthxRGBxsamples)
function mean_im = avg_image(im_tensor)
    imshape = size(im_tensor(:,:,:,1));
    mean_im = zeros(imshape, 'uint32');
    for img_ind=1:length(im_tensor(1,1,1,:))
        mean_im = mean_im + uint32(im_tensor(:,:,:,img_ind));
    end

    mean_im = double(mean_im) ./ length(im_tensor(1,1,1,:));
    %imshow(mean_im);
end