function generate_mod_lr_bic()
%% matlab code to genetate mod images, bicubic-downsampled lr, bicubic_upsampled images.

%% set parameters
% comment the unnecessary line
input_folder = 'E:\dataset\Youtube_hdr\test_hdr';
% save_mod_folder = '...';
save_LR_folder = 'E:\dataset\Youtube_hdr\test_hdr_bicx4';
%save_bic_folder = '...';

up_scale = 4;
mod_scale = up_scale;

if exist('save_mod_folder', 'var')
    if exist(save_mod_folder, 'dir')
        disp(['It will cover ', save_mod_folder]);
    else
        mkdir(save_mod_folder);
    end
end
if exist('save_LR_folder', 'var')
    if exist(save_LR_folder, 'dir')
        disp(['It will cover ', save_LR_folder]);
    else
        mkdir(save_LR_folder);
    end
end
if exist('save_bic_folder', 'var')
    if exist(save_bic_folder, 'dir')
        disp(['It will cover ', save_bic_folder]);
    else
        mkdir(save_bic_folder);
    end
end

idx = 0;
filepaths = dir(fullfile(input_folder,'*.*'));
for i = 1 : length(filepaths)
    [paths,imname,ext] = fileparts(filepaths(i).name);
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_rlt = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_rlt);
        % read image
        img = imread(fullfile(input_folder, [imname, ext]));
        type = class(img);
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        if exist('save_mod_folder', 'var')
            imwrite(img, fullfile(save_mod_folder, [imname, '.png']));
        end
        % LR
        im_LR = imresize(img, 1/up_scale, 'bicubic');
        if exist('save_LR_folder', 'var')
	    if strcmp(type, 'uint16')
	    	im_LR = uint16(round(im_LR*65535));
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '_bicx', num2str(up_scale), '.png']), 'bitdepth', 16);
	    else 
	    	imwrite(im_LR, fullfile(save_LR_folder, [imname, '_bicx', num2str(up_scale), '.png']));
	    end
	    %imwrite(im_LR, fullfile(save_LR_folder, [imname, '_bicx', num2str(up_scale), '.png']));
        end
        % Bicubic
        if exist('save_bic_folder', 'var')
            im_B = imresize(im_LR, up_scale, 'bicubic');
            imwrite(im_B, fullfile(save_bic_folder, [imname, '_bicx', num2str(up_scale), '.npy']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
