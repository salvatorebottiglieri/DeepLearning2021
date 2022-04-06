function [WGradient, BGradient]=backProp(net, X, T)
    WGradient = cell(1, net.numLayers);
    BGradient = cell(1, net.numLayers);
    delta = cell(1, net.numLayers);
    
    %Eseguo la forward propagation degli input
    net = forwardProp(net, X);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Calcolo dei delta per i nodi di output (ovvero le derivate della 
    %funzione di errore rispetto ai pesi). La formula utilizzata è valida
    %nel caso in cui la funzione di output sia la softmax, o la funzione identità,
    % e la funzione di errore sia la cross-entropy, o, rispettivamente, la 
    % somma dei quadrati: si sottrae componente per componente la
    %matrice dei target alla matrice di output, ottenendo la matrice dei
    %delta con tante righe quante sono le immagini del Set e tante colonne
    %quanti sono i nodi dello strato corrente (in questo caso l'ultimo)
    delta{end} = net.Z{end} - T; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    delta = calcola_delta(net,X,T);
    %Calcolo dei delta per i nodi interni
    for l = net.numLayers - 1 : -1 : 1
        delta{l} = delta{l+1} * net.W{l+1};

        %La seguente formula è valida nel caso in cui la funzione di output
        %dei nodi interni è la sigmoide
        delta{l} = (net.Z{l} .* (1-net.Z{l})) .* delta{l};  
    end
    
    %Calcolo dei gradienti
    Z = X;
    for l = 1 : net.numLayers
        WGradient{l} = delta{l}' * Z;
        BGradient{l} = sum(delta{l},1);
        Z = net.Z{l};
    end
end