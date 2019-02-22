
idx = randperm(100,10);
inp_vector = zeros(100,1);
inp_vector(idx) = normrnd(0,1.0,10,1);
std_dev = 0.005*norm(inp_vector)

h = ([1, 2, 3, 4, 3, 2, 1])'/16;
[ch,cw] = size(h);
%good that the vector is symmetric 
%dont need to inverse it while forming the A matrix

convolved = conv(inp_vector,h);
[h,w] = size(convolved);
noise = normrnd(0,std_dev,[h,w]);

y = convolved + noise;
[oh,ow] = size(y)

A = zeros(oh,100);
for i=1:100
    A(i:i+ch-1,i) = h;
end

lambda = std_dev;
eps=0.001*std_dev;
result = ista_c(y,A,lambda,eps);

rmse = norm(inp_vector - result)/norm(inp_vector)
%%
subplot(2,2,1);
stem(inp_vector)
title('Input')
subplot(2,2,2);
stem(y)
title('Noisy Input ')
subplot(2,2,3);
stem(result)
title('Reconstructed Output')
