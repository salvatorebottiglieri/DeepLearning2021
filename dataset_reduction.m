function ind_Dataset = dataset_reduction(Y, valPercent)
    if nargin < 2
        %Di default viene preso un mezzo del dataset
        valPercent = 0.5;
    end
    
    labels = unique(Y);
    ind_Dataset = [];
    for i = 1 : length(labels)
        c = labels(i); %c corrisponde all'i-esima label
        ind = find(Y == c); %gli elementi dell'array ind saranno gli indici degli elementi di Y pari a c
        N = length(ind);
        Nval = floor(valPercent*N); %arrotonda all'intero inferiore
        ind_Dataset = [ind_Dataset; ind(1 : Nval)];
    end
end

