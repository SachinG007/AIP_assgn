function psi = get_psi(theta)
    [cA,cH,cV,cD] = dwt2(theta,'db1');
    Avg = ( cH + cV + cD )/3;
    psi = [cA;Avg];

end
