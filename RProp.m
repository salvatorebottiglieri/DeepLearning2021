function net = RProp(net, WGradient, BGradient, eta_p, eta_m)

step_max = 50;
step_min = 0;

WStep = cell(1, net.numLayers);
BStep = cell(1, net.numLayers);
m1 = net.d;
for l = 1 : net.numLayers

    if not(isfield(net, 'oldWStep')) %Siamo alla prima chiamata della RProp
        m2 = length(net.B{l}); %Numero di nodi dello strato l
        %Gli step dello strato attuale vengono inizializzati casualmente
        WStep{l} = (0.2-0.05) .* rand(m2, m1) + 0.05;
        BStep{l} = (0.2-0.05) .* rand(1, m2) + 0.05;
        m1 = m2;
    else %Siamo in una chiamata della RProp successiva alla prima
        
        %Matrici dei prodotti tra i gradienti attuali e precedenti
        PWG = WGradient{l} .* net.oldWGradient{l};
        PBG = BGradient{l} .* net.oldBGradient{l};
        
        %Calcola i nuovi step dello strato attuale utilizzando le formule
        %previste dalla RProp
        WStep_tmp = net.oldWStep{l};
        WStep_tmp(PWG>0) = min(WStep_tmp(PWG>0) * eta_p, step_max);
        WStep_tmp(PWG<0) = max(WStep_tmp(PWG<0) * eta_m, step_min);
        WStep{l} = WStep_tmp;
        
        BStep_tmp = net.oldBStep{l};
        BStep_tmp(PBG>0) = min(BStep_tmp(PBG>0) * eta_p, step_max);
        BStep_tmp(PBG<0) = max(BStep_tmp(PBG<0) * eta_m, step_min);
        BStep{l} = BStep_tmp;
    end
    
    %Aggiorna i pesi della rete
    net.W{l} = net.W{l} - WStep{l} .* sign(WGradient{l});
    net.B{l} = net.B{l} - BStep{l} .* sign(BGradient{l});

end

%Aggiorniamo la rete
net.oldWStep = WStep;
net.oldBStep = BStep;
net.oldWGradient = WGradient;
net.oldBGradient = BGradient;

end

