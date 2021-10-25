function [X_TrainingSet, T_TrainingSet, X_ValidationSet, T_ValidationSet, X_TestSet, T_TestSet] = load_dataset()
    addpath ./mnist/loadMnist/

    %60000 images of size 28x28
    X = loadMNISTImages('mnist/train-images-idx3-ubyte');
    Y = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
    X = X';
    
    %Riduzione del numero di immagini ad un quarto del totale
    ind_Dataset = riduci_dataset(Y, 0.25);
    X = X(ind_Dataset, :);
    Y = Y(ind_Dataset, :);
    
    [ind_T,ind_V] = extractTsAndVs(Y);

    X_TrainingSet = X(ind_T, :); %data training
    Y_TrainingSet = Y(ind_T); %labels training

    X_ValidationSet = X(ind_V,:); %data Validation
    Y_ValidationSet = Y(ind_V); %labels validation

    %trasformo le label del training set in codifca binaria
    T_TrainingSet = zeros(size(X_TrainingSet, 1),10);
    for i = 1 : size(X_TrainingSet, 1)
        T_TrainingSet(i, Y_TrainingSet(i)+1)=1; %Target in codifica binaria
    end

    %trasformo le label del validation set in codifca binaria
    T_ValidationSet = zeros(size(X_ValidationSet,1),10);
    for i = 1 : size(X_ValidationSet)
        T_ValidationSet(i, Y_ValidationSet(i)+1) = 1; %Target in codifica binaria
    end

    %10000 images of size 28x28
    X = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
    Y = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
    X_TestSet = X';
    
    %Riduzione del numero di immagini
    ind_Dataset = riduci_dataset(Y);
    X_TestSet = X_TestSet(ind_Dataset, :);
    Y = Y(ind_Dataset, :);


    T_TestSet = zeros(size(X_TestSet,1), 10);
    for i = 1 : size(X_TestSet, 1)
        T_TestSet(i, Y(i)+1) = 1;
    end
end

