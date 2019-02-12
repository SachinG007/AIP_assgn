function ans = soft(y,lambda)

% if y>=lambda
%     ans = y-lambda;
%     
% elseif y<= -1 * lambda
%     ans = y + lambda;
%     
% else
%     ans = zeros(size(y));
ans = sign(y).*(max(0,abs(y)-lambda));
end

    
    
       

