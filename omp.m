function theta = omp(A,v_sub_snap,ep)
%ep = 2*2*8; %sigma*sqrt(m)
%ep = 130;
r = v_sub_snap;
r_norm = norm(r);
AT = [];
[~,w] = size(A);
support = zeros(1,w);
theta = zeros(w,1);
count = 0;
while r_norm.^2 > ep && count < 20%random num
    %Step 1
%     max_ind = 0;
%     max_val = 0;
    div = sqrt(sum(A.^2,1));
    [~,max_ind] = max(abs(A'*r)./div');
%     for j = 1:w
%         temp = abs(r'*A(:,j)/norm(A(:,j)));
%         if temp > max_val
%             max_ind = j;
%             max_val = temp;
%         end
%     end
    %max_ind
    %Step 2 
    if support(max_ind) == 1
        'error error errorerrorerrorerrorerrorerrorerrorerrorerrorerrorerrorerrorerrorerrorerrorerror'
        fprintf('Current Sparsity of reconstructed theta : %d.\n', nnz(support))
    end       
    support(max_ind) = 1;
    AT = [AT A(:,max_ind)];
    %Step 3
    size(pinv(AT)*v_sub_snap);
    nnz(support);
    theta(support>0) = pinv(AT)*v_sub_snap;
    %step 4
    r = v_sub_snap - AT*theta(support>0);
    r_norm = norm(r);
%     fprintf('Current norm of r : %d.\n', r_norm)
    count = count + 1;
end

nze = nnz(theta);
%fprintf('Sparsity of reconstructed theta : %d.\n', nze)

end