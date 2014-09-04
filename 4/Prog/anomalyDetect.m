function [ predict ] = anomalyDetect( model, Z )

constant = 4 * model.alpha' * makePolyKernel(model.supportVector, model.supportVector, model.c, model.d ) * model.alpha;
dataSize = size(Z, 1);
innerProduct = diag( makePolyKernel(Z, Z, model.c, model.d) );
kx = makePolyKernel(Z, model.supportVector, model.c, model.d) * model.alpha;
distance = innerProduct - 4 * kx + ones(dataSize,1) * constant;
predict = distance > model.r;

end

