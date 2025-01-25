set table "data/extrapolate.table"; set format "%.5f" 
set format "%.7e";;
f(x) = a * x ** 6 + b * x ** 5 + c * x ** 4 + d * x ** 3 + e * x ** 2 + f * x + g;
a = 1; b = 1; c = 1; d = 1; e = 1; f = 1; g = 1;
fit f(x) 'data/times.dat' using 1:2 via a,b,c,d,e,f,g;
plot[0:4500] f(x)/31536000000; 
