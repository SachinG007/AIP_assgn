% function [theta] = main3(x)
%     creating random sparse vector
% rand('seed',1)
    indices=randperm(100,10);
    x=zeros(100,1);
    x(indices) = normrnd(0,1.0,10,1);
    
    h = ([1,2,3,4,3,2,1]/16)';
    xmag = norm(x);
    c = conv(x,h);
    noise = normrnd(0,0.05*xmag,size(c));
    y = c+noise;
    
    phi = zeros(size(y,1),size(x,1));
    for i = 1:size(x,1)
        phi(i:i+size(h,1)-1,i) = h;
    end
    
    lambda = 1.0*(0.05*xmag);
    epsilon=0.00005*0.05*xmag;
    theta = istas(y, phi, lambda, epsilon);
%     display the reconstruction error
%     recerror  = norm(theta - x);
%     disp(recerror);
    rmse = norm(inp_vector - result)/norm(inp_vector)
figure()
stem(inp_vector)
figure()
stem(result)

