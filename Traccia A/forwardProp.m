function net = forwardProp(net, X)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Forward propagation per gli strati interni (nel nostro caso 1 solo)
    
    %X è la matrice di valori di input: ogni riga è una immagine e ogni
    %colonna rappresenta l'input da dare alla rete per quella immagine
    %(nodi di input per quella immagine)
    Z_prev = X;
    %Cicla per ogni strato interno della rete
    for l = 1 : (net.numLayers-1)
        %Calcola l'attivazione di ciascun nodo: per ogni l-esimo strato
        %calcola la matrice di attivazione data da tante righe quante sono
        %le immagini del Set e tante colonne quanti sono i nodi dello
        %strato corrente
        net.A{l} = Z_prev * net.W{l}' + net.B{l};
        %La funzione di output dei nodi interni è la sigmoide: restituisce
        %la matrice degli output dei nodi dello strato l
        %net.Z{l} = sigmoide(net.A{l});
        net.Z{l} = funzioneDiAttivazione(net.funAttivazione(l),net.A{l});
        Z_prev = net.Z{l};
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Forward propagation per lo strato di output
    
    %Matrice delle attivazioni
    net.A{end} = Z_prev * net.W{end}' + net.B{end};
    %La funzione di output per i nodi di output è la softmax: restituisce
    %la matrice degli output dei nodi dell'ultimo strato della rete
    net.Z{end} = softmax(net.A{end});
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

