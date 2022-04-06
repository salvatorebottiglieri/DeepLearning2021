epoche = zeros(1,10);

for i=1:10

    epoche(i) = test(80,[1],1.1,0.4);


end

media = mean(epoche);

disp("Array:");
disp(epoche);


disp("In media l'epoca migliore è:");
disp(media);

