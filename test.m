function Tm = test(NUM_HIDDEN_NODES,funAttivazione,eta_p,eta_m)
%clear all;
addpath ./mnist/loadMnist/
%alfa = input('alfa:  ');
%criterio_Stop = input('criterio_Stop:  ');
%nodi_Interni = input('N_nodiInterni:  ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRAINING E VALIDATION SET:

%60000 images of size 28x28
%X sarà una matrice di 60000 colonne e 784 righe: ogni colonna corrisponde
%ad un'immagine differente; ogni immagine (e quindi ogni colonna) è
%caratterizzata da 28x28 (784) righe (ovvero la loro dimensione)
X = loadMNISTImages('mnist/train-images-idx3-ubyte');
%Y è un vettore di 60000 label (una per ogni immagine): ovviamente possono
%esserci label uguali che si riferiscono ad una stessa immagine che compare
%in diverse colonne di X.
Y = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
%ind_T è il vettore degli indici che indicano la posizione delle immagini e
%delle label da considerare, rispettivamente in X e Y, per il TS;
%ind_V è il vettore degli indici che indicano la posizione delle immagini e
%delle label da considerare, rispettivamente in X e Y, per il VS
[ind_T,ind_V] = extractTsAndVs(Y);

%Trasponendo la matrice, si avrà che ogni riga corrisponderà ad una
%immagine; ogni immagine (quindi ogni riga) sarà caratterizzata da 784
%colonne (ovvero la dimensione di ogni riga)
X = X';

%%%%%%%%%%%%
%Per il TS
%Dalla matrice X si prendono solo le immagini individuate durante 
%l'estrazione (quindi le righe di X indicate da ind_T) e tutte le colonne 
X_TrainingSet = X(ind_T, :); %data training
%Dal vettore di label si prendono solo le label individuate durante
%l'estrazione (quindi solo size(ind_T) label di Y)
Y_TrainingSet = Y(ind_T); %labels training
%%%%%%%%%%%%

%%%%%%%%%%%%
%Per il VS 
X_ValidationSet = X(ind_V,:); %data Validation
Y_ValidationSet = Y(ind_V); %labels Validation
%%%%%%%%%%%%



%Le label del TS vengono trasformate in codifca binaria: la dimensione
%della matrice dei target sarà pari al numero di immagini del TS sulle
%righe e 10 sulle colonne (ovvero il numero di possibili classi
%restituibili)
T_TrainingSet = zeros(size(X_TrainingSet, 1), 10);
%Itera per la prima dimensione del TS, ovvero sul numero di immagini
for i = 1 : size(X_TrainingSet, 1)
    %Target in codifica binaria
    T_TrainingSet(i, Y_TrainingSet(i)+1) = 1;
    %Al termine ogni riga corrisponderà ad una immagine e per ogni immagine
    %la propria colonna sarà caratterizzata da tutti zeri e un solo 1 nella
    %j-esima colonna, ad indicare che l'i-esima immagine (ovvero l'i-esima
    %riga della matrice) corrisponde alla j-esima immagine tra le 10 note
end


%Le label del VS vengono trasformate in codifca binaria: la dimensione
%della matrice dei target sarà pari al numero di immagini del VS sulle
%righe e 10 sulle colonne (ovvero il numero di possibili classi
%restituibili)
T_ValidationSet = zeros(size(X_ValidationSet,1),10);
for i = 1 : size(X_ValidationSet)
    %Target in codifica binaria
    T_ValidationSet(i, Y_ValidationSet(i)+1)=1; 
    %Al termine ogni riga corrisponderà ad una immagine e per ogni immagine
    %la propria colonna sarà caratterizzata da tutti zeri e un solo 1 nella
    %j-esima colonna, ad indicare che l'i-esima immagine (ovvero l'i-esima
    %riga della matrice) corrisponde alla j-esima immagine tra le 10 note
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TEST SET:

%10000 images of size 28x28
%X sarà una matrice di 10000 colonne e 784 righe: ogni colonna corrisponde
%ad un'immagine differente; ogni immagine (e quindi ogni colonna) è
%caratterizzata da 28x28 (784) righe (ovvero la loro dimensione)
X = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
%Y è un vettore di 10000 label (una per ogni immagine): ovviamente possono
%esserci label uguali che si riferiscono ad una stessa immagine che compare
%in diverse colonne di X.
Y = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
%Trasponendo la matrice, si avrà che ogni riga corrisponderà ad una
%immagine; ogni immagine (quindi ogni riga) sarà caratterizzata da 784
%colonne (ovvero la dimensione di ogni riga)
X_TestSet = X';


%Le label del TE vengono trasformate in codifca binaria: la dimensione
%della matrice dei target sarà pari al numero di immagini del TE sulle
%righe e 10 sulle colonne (ovvero il numero di possibili classi
%restituibili)
T_TestSet = zeros(size(X_TestSet, 1), 10);
for i = 1 : size(X_TestSet, 1)
    %Target in codifica binaria
    T_TestSet(i, Y(i)+1) = 1;
    %Al termine ogni riga corrisponderà ad una immagine e per ogni immagine
    %la propria colonna sarà caratterizzata da tutti zeri e un solo 1 nella
    %j-esima colonna, ad indicare che l'i-esima immagine (ovvero l'i-esima
    %riga della matrice) corrisponde alla j-esima immagine tra le 10 note
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


MAX_EPOCHES = 200;
%Iperparametri:
step_max = 50;
step_min = 0;
alfa = 1;

CRITERIO_STOP = 0; %0 = GL; 1 = PQ

%La rete sarà caratterizzata da 1 strato di input, 1 strato di nodi interni
%e 1 strato di output.
%L'array passato come argomento avrà come valori: la dimensione dei nodi di
%input (28x28 = 784); il numero di nodi interni; il numero di possibili
%classi di output (10)
net = newNet([size(X_TrainingSet, 2) NUM_HIDDEN_NODES 10],funAttivazione);

%Numero di immagini del TS con cui addestrare la rete
N = size(X_TrainingSet, 1); 

%Array degli errori:
%Errore sul TrainingSet
Etr = zeros(1,MAX_EPOCHES);
%Errore su ValidationSet
Eva = zeros(1,MAX_EPOCHES);
%Miglior errore sul ValidationSet
Eopt = zeros(1,MAX_EPOCHES);

%Cicla sul numero totale di epoche
for numEpoch = 1 : MAX_EPOCHES
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %BATCH LEARNING. Lo scopo è minimizzare la funzione di errore e viene
    %fatto in due fasi:
    %1) Si applica la back propagation calcolando le derivate della
    %funzione di errore totale (TRATTANDOSI DI BATCH LEARNING E NON ONLINE: 
    %ALTRIMENTI SI SAREBBERO CALCOLATE LE DERIVATE DELLA FUNZIONE DI ERRORE 
    %AD OGNI INPUT E QUINDI ANCHE L'AGGIORNAMENTO SAREBBE AVVENUTO AD OGNI 
    %INPUT) rispetto ai pesi e tali derivate vengono usate per calcolare i
    %gradienti (sia per i pesi che per i bias);
    [WGradient,BGradient] = backProp(net, X_TrainingSet, T_TrainingSet);
    %2) I gradienti vengono usati per aggiornare i pesi effettivi della
    %rete, così da minimizzare effettivamente la funzione di errore:
    %l'aggiornamento dei pesi avviene utilizzando come procedura la RProp 
    net=RProp(net, WGradient, BGradient, eta_p, eta_m, step_max, step_min);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Si addestra la nuova rete in base alle modifiche apportate dalla
    %backProp e dalla RProp
    net = forwardProp(net, X_TrainingSet);
    %Calcola l'errore sul TS
    Etr(numEpoch) = cross_entropy(net.Z{end}, T_TrainingSet);
    
    %Si Valida la nuova rete in base alle modifiche apportate dalla
    %backProp e dalla RProp
    net = forwardProp(net, X_ValidationSet);
    %Calcola l'errore sul VS
    Eva(numEpoch) = cross_entropy(net.Z{end}, T_ValidationSet);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Salvataggio, ad ogni epoca, del miglior errore sul VS
    if numEpoch == 1
        Eopt(numEpoch) = Eva(numEpoch);
        %Tm sarà l'epoca in cui si ha il minor errore sul VS
        Tm = 1;
        %Si salva la miglior rete (quella col miglior errore sul VS)
        bestNet = net;
    else
        Eopt(numEpoch) = min(Eopt(numEpoch-1), Eva(numEpoch));
        if Eopt(numEpoch) < Eopt(numEpoch-1)
            Tm = numEpoch;
            bestNet = net;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Criteri di STOP
    GL = 100*(Eva(numEpoch)/Eopt(numEpoch) - 1);
    
    switch CRITERIO_STOP
        case 0 %GL 
            if GL > alfa
                break;
            end
            
        case 1 %PQ
            if mod(numEpoch, 5) == 0
            	%Dato dal rapporto tra la somma degli errori delle 5 epoche
            	%precedenti alla corrente diviso il minimo tra questi
            	%errori il quale è moltiplicato per 5 e al quale è
            	%sottratto 1. Tale rapporto è moltiplicato per 1000
                Pk = 1000 * ( sum(Etr(numEpoch-4:numEpoch))/(5*min( Etr(numEpoch-4:numEpoch))) -1 );
                PQ = GL / Pk;
                if PQ > alfa
                    break;
                end
            end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Stampe
    %Calcola accuratezza sul Validation Test
    accV = accuratezza(net.Z{end}, T_ValidationSet);
    disp(['numEpoch: ' num2str(numEpoch) ' Etr: ' num2str(Etr(numEpoch)) ' Eva: ' num2str(Eva(numEpoch)) ' Eopt: ' num2str(Eopt(numEpoch)) ' accV: ' num2str(accV) ]);
    
end            

%Sul Test Set si utilizza la miglior rete individuata ai passi precedenti
bestNet = forwardProp(bestNet, X_TestSet);
%Calcola l'accuracy sulla miglior rete trovata
accTest = accuratezza(bestNet.Z{end}, T_TestSet);
%Calcola errore sul TestSet
Ete = cross_entropy(bestNet.Z{end}, T_TestSet);

%Stampe
disp(['accTest: ' num2str(accTest)]);
disp(['Epoca_Di_Stop: ' num2str(numEpoch)]);
if CRITERIO_STOP == 0
    disp('Criterio_Di_Stop: GL');
else
    disp('Criterio_Di_Stop: PQ');
end
disp(['Ete: ' num2str(Ete)]);
disp(['Tm: ' num2str(Tm)]);

end
