n           = 128;
ms          = [40,50,64,80,100,120];
alpha       = 3;
avg_rmse    = zeros(6);

lambda  = diag(linspace(1,n,n).^(-alpha));
[U,~,~] = svd(randn(n));
sigma_x = U*lambda*U';
mu_x    = zeros(1,n);

for i=1:size(ms, 2)
    
    m   = ms(i);
    phi = normrnd(0, 1/m, [m,n]);
    xs  = mvnrnd(mu_x, sigma_x, 10)';
    
    sigma_noise = 0.01*mean(mean(abs(phi*xs)))^2;
    mu_noise    = zeros(1,m);
    
    for j=1:size(xs,2)
       
        x           = xs(:,j);        
        noise       = mvnrnd(mu_noise, sigma_noise*eye(m), 1)';
        y           = phi*x + noise;
        est_x       = (phi'*phi + sigma_x'/sigma_noise)\(phi'*y);
        avg_rmse(i) = avg_rmse(i) + sqrt(mean((est_x-x).^2));
        
    end
end

avg_rmse = avg_rmse/10;
plot(ms,avg_rmse);
xlabel('m');
ylabel('rmse');





