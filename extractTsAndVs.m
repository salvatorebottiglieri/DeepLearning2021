function [ind_T,ind_V] = extractTsAndVs(Y,valPercent)
    if nargin < 2
        valPercent = 0.33; %in questo modo il trainng set sarà il doppio del validation set
    end
    labels = unique(Y);
    
    ind_T = [];
    ind_V = [];
    for i = 1 : length(labels)
        c = labels(i); %c corrisponde all'i-esima label
        ind = find(Y == c); %gli elementi dell'array ind saranno gli indici degli elementi di Y pari a c
        N = length(ind);
        Nval = floor(valPercent*N); %arrotonda all'intero inferiore
        ind_V = [ind_V; ind(1 : Nval)];
        ind_T = [ind_T; ind(Nval+1 : end)];
    end
end