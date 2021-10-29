function net = forwardProp(net,X)
    %Forward propagation per gli strati interni
    Z_prev = X;
    for l=1:(net.numLayers-1)
        %Calcola l'attivazione di ciascun nodo
        net.A{l} = Z_prev * net.W{l}' + net.B{l};
        %La funzione di output dei nodi interni è la sigmoide
        net.Z{l} = sigmoide(net.A{l});
        Z_prev = net.Z{l};
    end
    
    %Forward propagation per lo strato di output
    net.A{end} = Z_prev * net.W{end}' + net.B{end};
    %La funzione di output per i nodi di output è la softmax
    net.Z{end} = softmax(net.A{end});
end

