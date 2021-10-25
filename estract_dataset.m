function [X_TrainingSet, T_TrainingSet, X_ValidationSet, T_ValidationSet, X_TestSet, T_TestSet] = estract_dataset()
    addpath ./mnist/loadMnist/

    %60000 images of size 28x28
    X = loadMNISTImages('mnist/train-images-idx3-ubyte');
    Y = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
    X = X';
    
    %images number reduction to 0.25 ratio 
    ind_Dataset = riduci_dataset(Y, 0.25);
    X = X(ind_Dataset, :);
    Y = Y(ind_Dataset, :);
    
    %extract both training and validation set in their indexes
    [ind_T,ind_V] = extract_TsAndVs(Y);

    X_TrainingSet = X(ind_T, :); %data training
    Y_TrainingSet = Y(ind_T); %labels training

    X_ValidationSet = X(ind_V,:); %data Validation
    Y_ValidationSet = Y(ind_V); %labels validation

    %convert the TS labels in binary code
    T_TrainingSet = zeros(size(X_TrainingSet, 1),10);
    for i = 1 : size(X_TrainingSet, 1)
        %binary code target
        T_TrainingSet(i, Y_TrainingSet(i)+1)=1; 
    end

    %convert the VS labels in binary code
    T_ValidationSet = zeros(size(X_ValidationSet,1),10);
    for i = 1 : size(X_ValidationSet)
        %binary code target
        T_ValidationSet(i, Y_ValidationSet(i)+1) = 1; 

    %10000 images of size 28x28
    X = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
    Y = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
    X_TestSet = X';
    
    %images number reduction
    ind_Dataset = riduci_dataset(Y);
    X_TestSet = X_TestSet(ind_Dataset, :);
    Y = Y(ind_Dataset, :);


    T_TestSet = zeros(size(X_TestSet,1), 10);
    for i = 1 : size(X_TestSet, 1)
        T_TestSet(i, Y(i)+1) = 1;
    end
end
