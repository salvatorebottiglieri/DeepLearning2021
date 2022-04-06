%Tale funzione si occuperà di estrarre l'indice di partenza del TS e del VS
function [ind_T,ind_V]=extractTsAndVs(Y,valPercent)
    %Se il numero di argomenti passati alla funzione è inferiore a 2, 
    %non è indicata la percentuale di distribuzione, la quale sarà settata
    %a 0.33
    if nargin < 2
        %valPercent indica la proporzione dei dati tra il TS e il VS: 
        %in questo caso il TS sarà sempre il doppio del VS
        valPercent=0.33; 
    end
    %Ogni label caratterizza un'immagine: vengono prese label senza
    %ripetizioni (una ad indicare tutte le immagini relative)
    labels=unique(Y);
    
    %Indice di partenza da cui estrarre i dati del TS
    ind_T=[];
    %Indice di partenza da cui estrarre i dati del TS
    ind_V=[];
    %Itero per ogni label
    for i=1:length(labels)
        %c corrisponde all'i-esima label
        c=labels(i);
        %ind è l'array degli indici delle label pari a c
        ind = find(Y==c);
        %N è la lunghezza dell'array di indici delle label pari a c
        N=length(ind);
        %floor arrotonda all'intero inferiore
        Nval = floor(valPercent*N); 
        ind_V=[ind_V; ind(1:Nval)];
        ind_T=[ind_T; ind(Nval+1:end)];
    end
end