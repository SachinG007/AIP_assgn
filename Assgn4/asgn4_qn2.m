rmse1_store = [];
rmse2_store = [];
sigma_store = [];

% sig_mag_arr = [0.01 0.05 0.1 0.3 0.5];
% sparsity_arr = [5 10 15 20 25 30];
% k_arr = [0.5 1 1.5 2 2.5 3];

for i = 1:1
    i
    
%     k = k_arr(i);

    %HyperParams
    k = 1;
    sparsity = 10;
    sig_mag = 0.2;

    %Produce f1
    U = dctmtx(256);
    s = zeros(64*4,1);
    idx = randi(64*4,sparsity,1);
    s(idx) = 1;
    f1 = U * s;
    figure
    title('RMSE Sigma = 0.2 * avg(f1+f2) Sparsity = 10 k=1');
    subplot(2,3,1)
    plot(f1);
    title('Origial Signal f1');

    %Produce f2
    W = k * eye(256);
    idx = randi(64*4,sparsity,1);
    s(idx) = 1;
    f2 = W * s;
    subplot(2,3,2)
    plot(f2);
    title('Origial Signal f2');

    %Produce noise 
    sigma = sig_mag * mean(f1+f2);
    sigma_store = [sigma_store sigma]; 
    noise = normrnd(0,sigma^2,[256,1]);

    f = f1+f2+noise;
    subplot(2,3,3)
    plot(f);
    title('Origial Signal f');
    

    %Reconstruction 
    A = [U W];
    %initial guess
    rec_f0 = A'*f;
    rec_f_basis = l1eq_pd(rec_f0, A, [], f, 1e-3);
    rec_f_basis1 = rec_f_basis(1:256,:);
    rec_f_basis2 = rec_f_basis(257:512,:);

    rec_f1 = U * rec_f_basis1;
    rec_f2 = W * rec_f_basis2;
    subplot(2,3,4)
    plot(rec_f1)
    title('Reconstructed Signal f1');
    subplot(2,3,5)
    plot(rec_f2)
    title('Reconstructed Signal f2');
    subplot(2,3,6)
    plot(rec_f1+rec_f2)
    title('Reconstructed Signal f');

    %%
    %RMSE Cal
    rmse1 = norm(rec_f1 - f1)/norm(f1);
    rmse2 = norm(rec_f2 - f2)/norm(f2);
    rmse1_store = [rmse1_store rmse1];
    rmse2_store = [rmse2_store rmse2];
    
end
%%
% figure
% plot(k_arr,rmse1_store,k_arr ,rmse2_store)
% title('RMSE Sigma = 0.2 * avg(f1+f2) Sparsity = 10');
% xlabel('k') ;
% ylabel('RMSE Vals '); 
% legend({'RMSE f1','RMSE f2'},'Location','northwest')