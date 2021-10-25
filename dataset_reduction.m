function new_dataset  = dataset_reduction(old_dataset,reduction_factor)
%DATASET_REDUCTION reduce the size of dataset
if nargin < 2
    reduction_factor = 0.75;
end

labels = unique(old_dataset);
new_dataset = [];
for i = 1: length(labels)
    c = labels(i);
    indexes = find(old_dataset == c);
    indexes_size = length(indexes);
    n_value = floor(indexes_size*reduction_factor);
    new_dataset = [old_dataset;ind(1:n_value)];
end


end

