function [bestNet, epoca_stop, Tm] = trainNet(net, X_TrainingSet, T_TrainingSet, X_ValidationSet, T_ValidationSet, eta_p, eta_m, criterio_stop, alfa)
    if nargin < 8
        criterio_stop = -1;
    end

    MAX_EPOCHES = 200;
    %Array di errori sul TrainingSet
    Etr = zeros(1,MAX_EPOCHES);
    %Array di errori sul ValidationSet
    Eva = zeros(1,MAX_EPOCHES);
    
    for numEpoch = 1 : MAX_EPOCHES

        %BATCH LEARNING
        [WGradient,BGradient] = backProp(net,X_TrainingSet, T_TrainingSet);
        net = RProp(net, WGradient, BGradient, eta_p, eta_m);

        net = forwardProp(net, X_TrainingSet);
        %Calcolare l'errore sul Traning
        Etr(numEpoch) = cross_entropy(net.Z{end}, T_TrainingSet);

        net = forwardProp(net, X_ValidationSet);
        %Calcolare l'errore sul Validation
        Eva(numEpoch) = cross_entropy(net.Z{end}, T_ValidationSet);
        %Calcolare l'errore ottimo e la miglior rete trovata
        if numEpoch == 1
            Eopt = Eva(numEpoch);
            bestNet = net;
            Tm = 1;
        else
            if Eva(numEpoch) < Eopt
                Eopt = Eva(numEpoch);
                bestNet = net;
                Tm = numEpoch;
            end
        end
        
        %stampe
        %accV = accuratezza(net.Z{end}, T_ValidationSet);
        %disp(['numEpoch: ' num2str(numEpoch) ' Etr: ' num2str(Etr(numEpoch)) ' Eva: ' num2str(Eva(numEpoch)) ' Eopt: ' num2str(Eopt) ' accV: ' num2str(accV) ]);

        GL = 100*(Eva(numEpoch)/Eopt - 1);

        %Criteri di STOP
        switch criterio_stop
            case 0 %GL 
                if GL > alfa
                    break;
                end

            case 1 %PQ
                if mod(numEpoch, 5) == 0
                    Pk = 1000 * ( sum(Etr(numEpoch-4:numEpoch))/(5*min( Etr(numEpoch-4:numEpoch))) -1 );
                    PQ = GL / Pk;
                    if PQ > alfa
                        break;
                    end
                end
        end

    end
    epoca_stop = numEpoch;

end

