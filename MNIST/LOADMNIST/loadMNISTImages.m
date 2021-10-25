function images_res = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

%Ridimensiona ogni immagine in una matrice 10x10
images_res = imresize(images, [10 10]);
% Reshape to #pixels x #examples
images_res = reshape(images_res, size(images_res, 1) * size(images_res, 2), size(images_res, 3));
% Convert to double and rescale to [0,1]
images_res = double(images_res) / 255;

end
