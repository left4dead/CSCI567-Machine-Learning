function [ model ] = anomalyTrain( data, d, C )

constant = 1;

K = makePolyKernel(data, data, constant, d);
f = -0.25 * diag(K);

dataSize = size(data, 1);

A = ones(1, dataSize);
b = 0.5;

opts = optimset('Algorithm','interior-point-convex', 'display', 'off', 'MaxIter', 1000);

alpha = quadprog(K, f, [], [], A, b, zeros(1, dataSize), C * ones(1, dataSize), [], opts);
indices = alpha > 0.00001;
supportVector = data(indices,:);
param = alpha(indices);
index = 1;
while index<size(param,1) && param(index)>= C*(1-0.01) 
    index= index + 1;
end

r = makePolyKernel(supportVector(index,:), supportVector(index,:), constant, d) - 4 * makePolyKernel(supportVector(index,:), supportVector, constant, d) * param + 4 * param' *  makePolyKernel(supportVector, supportVector, constant, d) * param;
    

model = struct('supportVector', supportVector, 'alpha' , param, 'r', r, 'c', constant, 'd', d);

end





