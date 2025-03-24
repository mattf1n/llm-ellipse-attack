set table "overleaf/data/fit.table"; set format "%.5f" 
set format "%.7e";;
f(x) = a * x ** 6 + b;
a = 1; b = 1; c = 1; d = 1; e = 1; f = 1; g = 1;
fit f(x) 'overleaf/data/times-mosek.dat' using 1:2 via a,b;
plot[0:210] f(x)/60; 
