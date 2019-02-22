function theta = ista_c_haar(y,phi)

%random init for k=0
theta = normrnd(0,1,[64,1]);
%get the alpha
alpha = 178.0;
% eig_vals = eig(A'*A);
% alpha = max(eig_vals)+1;

diff = 10; %rand value
c_norm = norm(theta);
%start loop 
% for k=1:10
count = 0;
while diff > 0.05
    term_2 = 10/(2*alpha);
    at = phi*get_psi(theta);
    at2 = y-at;
    at3 = phi'*at2;
    at4 = get_psi(at3);
    term_1 = theta + (1/alpha)*at4;
    theta = soft(term_1,term_2);
    diff = norm(theta) - c_norm
    c_norm = norm(theta);
    count = count + 1;

end
% count
end

