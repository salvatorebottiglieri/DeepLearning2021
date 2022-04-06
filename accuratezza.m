function acc=accuratezza(Y,T)

%%%%%%%
%QUI CONSIDERO COME REGOLA DI DECISIONE QUELLA DI SCEGLIERE LA CLASSE CON
%PROBABILITA' MASSIMA, DATO L'INPUT
%%%%%%%%%

%Y e T sono matrici le cui i-esime righe rappresentano rispettivamente
%l'output e i target rispetto all'i-esimo input. Per ogni riga la j-esima 
%colonna rappresenta la probabilità che dato l'input j sia la sua classe di
%appartenenza.

%Per ogni riga, e quindi ogni output, si estrae l'indice del valore massimo 
%: quindi l'indice della probabilità massima e quindi l'indice che 
%corrisponde al risultato più plausibile per l'i-esimo input
[~, ind_Y] = max(Y, [], 2);
%Lo stesso è fatto per i target
[~, ind_T] = max(T, [], 2);
%Confrontando gli indici si verifica se l'i-esimo risultato individuato
%corrisponde a quello atteso. Si sommano tutti gli indici così ottenendo in
%"correct" la somma dei valori accertati essere corretti
correct = sum(ind_Y == ind_T);
%A questo punto si calcola l'accuratezza
acc = correct/length(ind_Y);
end