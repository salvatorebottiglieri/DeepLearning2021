
fid = fopen('test_iperparametri.csv', 'w');
fprintf(fid, '%s, %s, %s, %s\n', 'Numero Nodi interni', 'Eta+', 'Eta-', 'Accuracy');


accuracy = zeros(1,10);
nodi = [10 30 50 70 90];
etap = [1.1 1.2 1.3 1.4];
etam = [0.3 0.4 0.5 0.6];

disp("Numero nodi Eta+ Eta- Accuracy_media");

for i=1:length(nodi) % cicla sui nodi

    for j=1:length(etap) % cicla su etap

        for k=1:length(etam) %cicla su etam

            for p=1:10 %calcola media

                accuracy(p) = test(nodi(i),[1], etap(j),etam(k));
            end

             media = mean(accuracy);
             disp("STAMPO SUL FILE:")
             disp(nodi(i));
             disp(etap(j));
             disp(etam(k));
             disp(media);
          
             fprintf(fid, '%d, %f,%f,%.4f\n', nodi(i), etap(j),etam(k),media);


        end

    end
end



fclose(fid);
