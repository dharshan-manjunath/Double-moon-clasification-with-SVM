function G = mysigmoid(U,V)
 %Sigmoid kernel function with slope gamma and intercept c
gamma = 100;
c = -10;
G = tanh(gamma*U*V' + c);
end