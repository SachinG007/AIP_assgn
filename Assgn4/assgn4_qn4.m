% rng(1)
clear
sparsity = 5;
m = 70;
p = 256;
frac = .5;

%Produce f
U = dctmtx(p);
%sparse codes
s = zeros(64*4,1);
idx = randi(64*4,sparsity,1);
s(idx) = 1;
f = U * s;

%sensing matrix same as qn1
phi = randi(2,m,p);
phi(phi==2) = -1;
phi = phi/sqrt(m);

sigma = 100 * mean(f);
noise = normrnd(0,sigma^2,[m,1]);
    
%Ideal measured values
y_ideal = phi*f + noise;
y_actual = y_ideal;

%Incorporate the saturation 
y_sorted = sort(y_actual,'descend')
Lidx = floor(frac*m);
if Lidx == 0
    Lidx =1;
end
L = y_sorted(Lidx);
y_actual(y_actual>= L) = L;

I = find(y_actual<L);
y_new = y_actual(I);
phi_new = phi(I,:);

A = phi_new * U;
A_org = phi*U;

s_rec = OMP2( A, y_new, 10);
s_rec_nonsat = OMP2(A_org, y_ideal,sparsity);
f_rec = U * s_rec;
f_rec_nonsat = U * s_rec_nonsat ;

rmse_sat = norm(f_rec - f)/norm(f)
rmse_nonsat = norm(f_rec_nonsat - f)/norm(f)