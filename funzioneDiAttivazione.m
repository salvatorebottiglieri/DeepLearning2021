function Y = funzioneDiAttivazione(funzione, inputFunzione)
%handler che richiama una determinata funzione di attivazione in base al
%valore del parametro "funzione", passandogli il valore "inputFunzione".

switch funzione

    case 1
        Y =  sigmoide(inputFunzione);

    case 2
        Y = softmax(inputFunzione);

    case 3
        Y = identity(inputFunzione);

    otherwise
        Y = sigmoide(inputFunzione);
end



