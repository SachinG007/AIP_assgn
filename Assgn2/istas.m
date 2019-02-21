function [theta] = ista(Y, A, lambda, epsilon)
[rowY,colY] = size(Y);
[rowA,colA] = size(A);
theta = randn(colA, colY);
e = eig(A.'*A);
alpha = max(e)+1.0;
thr = lambda/(2*alpha);
% niter = 50;
% basic objective/loss function
% for i=1:niter
%     soft1 = theta + (1/alpha)*A.'*(Y-A*theta);
%     theta = wthresh(soft1,'s',thr);
% %     diff = Y-A*theta;
% %     J = norm(diff).^2+lambda*norm(theta,1);
% %     loss(i) = J;
% end
count=0;
soft1 = theta + (1/alpha)*A.'*(Y-A*theta);
theta_prev = wthresh(soft1,'s',thr);

while(norm(theta-theta_prev)>= epsilon)
    count=count+1;
    theta_prev = theta;
    soft1 = theta + (1/alpha)*A.'*(Y-A*theta);
    theta = wthresh(soft1,'s',thr);
end
% disp(count);
% x = [1:niter];
% y = loss;
% figure, plot(x,y);
end
