function output = GHI_Kernel(u, v)
    beta = 2;
    u=u.^beta;
    v=v.^beta;
    test = min(u,v);
    output = sum(test,2);
end