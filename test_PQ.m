fid = fopen('risultati_test_PQ.csv', 'a+');
fprintf(fid, '%s, %s, %s, %s, %s\n', 'Alfa', 'Accuracy', 'Errore Test Set', 'Epoca della miglior rete', 'Epoca di stop');
NUM_TEST = 10;
NUM_NODI_INTERNI = 80;
ETA_P = 1.1;
ETA_M = 0.5;
CRITERIO_STOP = 1; %0 = GL; 1 = PQ
ALFA = [0.5 0.75 1];

%Carica il dataset distribuendolo tra Training, Validation e Test Set
[X_TrainingSet, T_TrainingSet, X_ValidationSet, T_ValidationSet, X_TestSet, T_TestSet] = load_dataset();
    
for a = ALFA
    accTest = zeros(1, NUM_TEST);
    Ete = zeros(1, NUM_TEST);
    best_epoca = zeros(1, NUM_TEST);
    stop = zeros(1, NUM_TEST);
    disp(['Alfa ' num2str(a)]);
    for i = 1 : NUM_TEST
        disp(['Test numero...' num2str(i)]);

        %Crea una nuova rete con un opportuno numero di input, un solo strato 
        %interno con il numero di nodi stabilito e 10 nodi di output (uno per
        %ogni possibile classe)
        net = newNet([size(X_TrainingSet, 2) NUM_NODI_INTERNI 10]);

        %Esegue un addestramento della rete con i parametri stabiliti per il
        %criterio di stop
        [bestNet, epoca_stop, Tm] = trainNet(net, X_TrainingSet, T_TrainingSet, X_ValidationSet, T_ValidationSet, ETA_P, ETA_M, CRITERIO_STOP, a);
        
        %Lancia sul Test Set la miglior rete individuata durante l'addestramento
        bestNet = forwardProp(bestNet, X_TestSet);
        %Calcolo accuratezze per ogni ripetizione
        accTest(i) = accuratezza(bestNet.Z{end}, T_TestSet);
        %Calcolo errori sul test set per ogni ripetizione
        Ete(i) = cross_entropy(bestNet.Z{end}, T_TestSet);
        %Calcolo delle epoche in cui è stata individuata la miglior rete
        %per ogni ripetizione
        best_epoca(i) = Tm;
        %Calcolo dell'epoca di stop per le ripetizioni
        stop(i) = epoca_stop;
        
        disp(['accTest: ' num2str(accTest(i)) ' Ete: ' num2str(Ete(i)) ' best_epoca: ' num2str(best_epoca(i)) ' stop: ' num2str(stop(i))]);
    end

    %Calcolo accuratezza media
    accuracy = media(accTest);
    %Calcolo errore medio sul Test Set
    errore_TestSet = media(Ete);
    %Calcolo epoca media in cui è stata individuata la miglior rete
    epoca_migliore = round(media(best_epoca));
    %Calcolo l'epoca media di stop 
    epoca_stop = round(media(stop));
    

    %Stampe
    disp('Fine test.');
    fprintf(fid, '%.2f, %.4f, %.4f, %d, %d\n', a, accuracy, errore_TestSet, epoca_migliore, epoca_stop);
end

fclose(fid);



