%numNeurons è un vettore che contiene il numero di input in prima
%posizione, nelle successive il numero di nodi per ogni strato
function net = newNet(numNeurons)
d = numNeurons(1);
%len è il numero di strati della rete
len = length(numNeurons) - 1;

%W(i) è la matrice dei pesi dell'i-esimo strato 
W=cell(1,len);
%B(i) è l'array di bias dell'i-esimo strato
B=cell(1,len);
%A(i) è la matrice delle attivazione dei nodi dell'i-esimo strato
A=cell(1,len);
%Z(i) è la matrice di output dei nodi dell'i-esimo strato
Z=cell(1,len);

m1=d;
for i=1:len
    m2=numNeurons(i+1);
    
    %La matrice dei pesi ha una riga per ogni nodo dello strato corrente,
    %contenente i pesi degli archi entranti in quel nodo (uno per ogni nodo
    %dello strato precedente)
    %I pesi sono inizializzati con valori compresi tra -0.5 e 0.5
    W{i}=rand(m2, m1) - 0.5;
    
    %Il vettore dei bias ha un elemento per ogni nodo dello strato
    %I bias sono inizializzati con valori compresi tra -0.5 e 0.5
    B{i}=rand(1, m2) - 0.5;
    
    m1=m2;
end

%Aggiornamento della rete
net.d=d;
net.numLayers=len;
net.W=W;
net.B=B;
net.A=A;
net.Z=Z;
end

