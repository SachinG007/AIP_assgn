function theta = ista_c(y,A,lambda,eps)

%random init for k=0
[yh,yw] = size(y);
theta = normrnd(0,1.0,[100,1]);
%get the alpha
eig_vals = eig(A'*A);
alpha = max(eig_vals);

diff = 10; %rand value
c_norm = 0%norm(theta);
%start loop 
% for k=1:10
count = 0;
while diff > eps & (count < 10000)
    term_2 = 1/(2*alpha);
    term_1 = theta + (1/alpha)*A'*(y-A*theta);
    theta = soft(term_1,term_2);
    diff = abs(norm(theta) - c_norm)
%     norm(theta)
    c_norm = norm(theta);
    count = count + 1

end
count;
end

