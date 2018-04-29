function [nmi]=NMI(label_a, label_b)
    if(length(label_a)~=length(label_b))
        nmi=0;
        return;
    end
    com_a=unique(label_a);
    com_b=unique(label_b);
    
    norm_a=0;
    norm_b=0;
    n=length(label_a);
    
    for h=1:length(com_a)
       n_a=length(find(label_a==com_a(h)));
       norm_a=norm_a+n_a*log(n_a/n);
    end
    
    for l=1:length(com_b)
        n_b=length(find(label_b==com_b(l)));
        norm_b=norm_b+n_b*log(n_b/n);
    end
    
   
    numerator=0;
   
    for h=1:length(com_a)
        n_a=length(find(label_a==com_a(h)));
        for l=1:length(com_b)
            n_b=length(find(label_b==com_b(l)));
            
            n_ab=0;
            for i=1:n
                if(label_a(i)==com_a(h)&&label_b(i)==com_b(l))
                    n_ab=n_ab+1;
                end
            end
            if(n_ab~=0)
                 numerator=numerator+n_ab*log((n*n_ab)/(n_a*n_b));
            end
        end
    end
    nmi=numerator/(sqrt(norm_a*norm_b));
end