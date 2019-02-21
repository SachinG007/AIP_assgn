function ans = soft(term1,lambda)
% var = zeros(64,1);
% for i=1:64
%     
%     if term1(i)>=lambda
%         var(i) = term1(i)-lambda;
% 
%     elseif term1(i)<= -1 * lambda
%         var(i) = term1(i) + lambda;
% 
%     else
%         var(i) = 0;
%     end
% end
% 
% ans = var
ans = sign(term1).*(max(0,abs(term1)-lambda));
end

    
    
       

