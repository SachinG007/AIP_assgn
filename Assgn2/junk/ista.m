function theta = ista(y,A)

%random init for k=0
theta = normrnd(0,1,[64,1]);
%get the alpha
eig_vals = eig(A'*A);
alpha = max(eig_vals)+1;

diff = 10; %rand value
c_norm = norm(theta);
%start loop 
% for k=1:10
count = 0;
while diff > 0.1
    term_2 = 8/(2*alpha);
    term_1 = theta + (1/alpha)*A'*(y-A*theta);
    theta = soft(term_1,term_2);
    diff = norm(theta) - c_norm;
    c_norm = norm(theta);
    count = count + 1;

end
% count
end

