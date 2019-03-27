%Hyperparams
N = 600;
Nt = 100;
p = 16*16;
K = 128;
f = 0.01;
m = 180+40;
[imgs labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', N,100);
[test_imgs test_labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', Nt,700);
% for k=1:9
%     sum(labels==k)
% end
imgs = imresize(imgs, [16 16]);
imgs = reshape(imgs,256,N);
test_imgs = imresize(test_imgs, [16 16]);
test_imgs = reshape(test_imgs,256,Nt);
% a = reshape(imgs(:,4500),16,16)
% imshow(a)
%%
%generate sensing matrices
phi = randi(2,N,m,p);
phi(phi==2) = -1;
phi = phi/sqrt(m);

phi_concat = zeros(m,p*N);
for k = 1:N
    phi_concat(:,(k-1)*p+1:k*p) = reshape(phi(k,:,:),m,p);
end


temp = 0;
for k = 1:N
    phi_t = reshape(phi(k,:,:),m,p);
    temp = temp + sum(phi_t*imgs(:,k));
end
sigma = f * (1/(m*N))*temp;

y = zeros(m,N);
for k = 1:N
    noise = normrnd(0,sigma^2,[m,1]);
    phi_t = reshape(phi(k,:,:),m,p);
    y(:,k) = phi_t*imgs(:,k) + noise;
end

y_test = zeros(m,Nt);
for k = 1:Nt
    noise = normrnd(0,sigma^2,[m,1]);
    phi_t = reshape(phi(k,:,:),m,p);
    y_test(:,k) = phi_t*test_imgs(:,k) + noise;
end

done = 1


Dr = normrnd(0,1,[p,K]);
%%
for iter = 1:20
    iter
    
    %OMP for Coefficients estimate
    sparse_codes = zeros(K,N);
    for k = 1:N
%         phi_t = reshape(phi(k,:,:),m,p);
        phi_t = phi_concat(:,(k-1)*p+1:k*p);
        sparse_codes(:,k) = OMP2(phi_t*Dr,y(:,k),40);
    end

    %Dictionary atom update
    %for every dict column
    for k = 1:K
        
        
        rhs_term = zeros(p,1);
        norm_term = zeros(p,p);
        yik_store = zeros(m,N);
        
        for i = 1:N

            %get yik (yi_mius_k
            d_temp = zeros(p,1);
            for l = 1:K
                
                if l~=k            
                    d_temp = d_temp + Dr(:,l)*sparse_codes(l,i);
                end
            end
            
%             phi_t = reshape(phi(i,:,:),m,p);
            phi_t = phi_concat(:,(i-1)*p+1:i*p);
            yik = y(:,i) - phi_t*d_temp;
            yik_store(:,i) = yik;
            yikt = yik * sparse_codes(k,i);

            rhs_term = rhs_term + phi_t'*yikt;
            norm_term = norm_term + phi_t'*phi_t*sparse_codes(k,i)^2;

        end
        
        k
        Dr(:,k) = inv(norm_term)*rhs_term;
        Dr(:,k) = Dr(:,k)/norm(Dr(:,k));
        
        for i = 1:N
            phi_t = phi_concat(:,(i-1)*p+1:i*p);
            num = Dr(:,k)'*phi_t'*yik_store(:,i);
            denom = Dr(:,k)'*phi_t'*phi_t*Dr(:,k);
            sparse_codes(k,i) = num/denom;
        end
            
    end
    done = 1
    
rec_signals = Dr * sparse_codes;
rmse = 0;
for k = 1:N
    rmse = rmse + norm(rec_signals(:,k) - imgs(:,k))/norm(imgs(:,k));
end
rmse = rmse/N

% Dr_img = reshape(Dr,16,16,1,K);
% imgs = reshape(imgs,16,16,1,N);
% montage(Dr_img)

end

%%
Dr_img = reshape(Dr,16,16,1,K);
Dr_img = Dr_img * 2;
% imgs = reshape(imgs,16,16,1,N);
montage(Dr_img)
% done =1

% img = reshape(rec_signals(:,100),16,16);
% imshow(img)

%%
%testing
sparse_codes_test = zeros(K,Nt);
for k = 1:Nt
%         phi_t = reshape(phi(k,:,:),m,p);
    phi_t = phi_concat(:,(k-1)*p+1:k*p);
    sparse_codes_test(:,k) = OMP2(phi_t*Dr,y_test(:,k),40);
end
rec_signals_test = Dr * sparse_codes_test;

test_img_store = zeros(16,16,1,20);
for k = 1:20
    test_img_store(:,:,:,k) = reshape(rec_signals_test(:,2*k),16,16,1);
end
% imshow(test_img)
montage(test_img_store)
rmse = 0;
for k = 1:Nt
    rmse = rmse + norm(rec_signals_test(:,k) - test_imgs(:,k))/norm(test_imgs(:,k));
end
rmse = rmse/Nt

%%
%Comparison with 2d DCT
U = dctmtx(p);
rmse_dct = 0;
for k = 1:Nt
    phi_t = reshape(phi(k,:,:),m,p);
    A = phi_t*U;
    img_rec_dct = U * OMP2( A, y_test(:,k), 40);
    rmse_dct = rmse_dct + norm(img_rec_dct - test_imgs(:,k))/norm(test_imgs(:,k));
end
rmse = rmse_dct/Nt
    