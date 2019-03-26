rmse_store = [];
m_arr = [30, 40, 50, 70, 90];

for it = 1:5
    
    it 
    
    %Hyperparams
    N = 200;
    p = 100;
    K = 20;
    f = 0.01;
    % m = 50;
    m = m_arr(it);

    D = normrnd(0,1,[p,K]);
    for k = 1:K
        D(:,k) = D(:,k)/norm(D(:,k));
    end

    S = zeros(K,N);
    for k = 1:N
        idx = randi(K,4,1);
        S(idx,k) = randi(10,4,1);
    end

    sig = D*S;

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
        temp = temp + sum(phi_t*sig(:,k));
    end
    sigma = f * (1/(m*N))*temp;

    y = zeros(m,N);
    for k = 1:N
        noise = normrnd(0,sigma^2,[m,1]);
        phi_t = reshape(phi(k,:,:),m,p);
        y(:,k) = phi_t*sig(:,k) + noise;
    end


    Dr = normrnd(0,1,[p,K]);

    %%
    for iter = 1:50
        iter

        %OMP for Coefficients estimate
        sparse_codes = zeros(K,N);
        for k = 1:N
%             phi_t = reshape(phi(k,:,:),m,p);
            phi_t = phi_concat(:,(k-1)*p+1:k*p);
            sparse_codes(:,k) = OMP2(phi_t*Dr,y(:,k),10);
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
%                 phi_t = reshape(phi(i,:,:),m,p);
                phi_t = phi_concat(:,(i-1)*p+1:i*p);
                yik = y(:,i) - phi_t*d_temp;
                yik_store(:,i) = yik;
                yikt = yik * sparse_codes(k,i);

                rhs_term = rhs_term + phi_t'*yikt;
                norm_term = norm_term + phi_t'*phi_t*sparse_codes(k,i)^2;

            end
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
            rmse = rmse + norm(rec_signals(:,k) - sig(:,k))/norm(sig(:,k));
        end
        rmse = rmse/N

    end


    rec_signals = Dr * sparse_codes;
    rmse = 0;
    for k = 1:N
        rmse = rmse + norm(rec_signals(:,k) - sig(:,k))/norm(sig(:,k));
    end
    rmse = rmse/N
    rmse_store = [rmse_store  rmse];
    
end