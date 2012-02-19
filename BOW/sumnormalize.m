function x=sumnormalize(x,dim)

if dim==1
    
    [m,n]=size(x);
    for i=1:n
        sumc(:,i)=sum(x(:,i));
    end
    
    for i=1:n
        for j=1:m
            x(j,i)= x(j,i)/sumc(i);
        end
      
    end
    
end





end