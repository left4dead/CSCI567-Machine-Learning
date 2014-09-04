function W = run_1_logreg()

Data = importdata('HW3-USPS-split.mat');

features = Data.X.train;
labels = Data.y.train;
C = Data.C;
Lambda = [90000;70000;120000;30000;120000;30000;10000;160000;70000;10000];
featureSize = size(features, 1) +1;
W = zeros(featureSize, C);

for i = 1:C
	fprintf('One Vesus Rest on class %d \n', i);
    trainingLabels = (labels == i);
    bestW = logreg_DISC ( features, trainingLabels, Lambda(i));
    W(:,i) = bestW;
end


end



function [ oldW ] = logreg_DISC ( features, labels, lambda )

dataSize = size(features,2);
features = [ones(1,dataSize); features];
featureSize = size(features, 1);
stepSize = 1;
bound = (0.5).^15;

logLoss = zeros(100,1);

oldW = rand(featureSize,1)*0.001 - ones(featureSize,1) * 0.0005;
oldLoss = getError(features, labels, oldW, lambda);
for i = 1:100    
    logLoss(i) = oldLoss;
	stepSize = 1;
    [J, H] = derivation (features, labels, oldW, lambda);
    newW = oldW - stepSize * (H\J);
    newLoss = getError(features, labels, newW, lambda);
    while(newLoss >= oldLoss && stepSize >= bound)
        stepSize = stepSize /2;
        newW = oldW - stepSize * (H\J);
        newLoss = getError(features, labels, newW, lambda);
    end
    if newLoss >= oldLoss
    	break;
    end
    oldW = newW;
    oldLoss = newLoss;
end


end

function [ J, H ] = derivation ( features, labels, W, lambda )

WT = W';
featureSize = size(features, 1);
J = zeros(featureSize, 1);
H = zeros(featureSize, featureSize);

dataSize = size(features, 2);

for i = 1:dataSize
    sigmoid = 1 ./(1+exp(WT * features(:,i) * -1));
    J = J + ( (sigmoid-labels(i)) * features(:,i) );
    H = H + sigmoid * (1-sigmoid) * features(:,i) * (features(:,i))';
end


%regularization

I = eye(featureSize);
I(1,1) = 0;
J = J + lambda * I * W;
H = H + lambda * I;

end

function [loss] = getError (features, labels, W, lambda)

WT = W';
dataSize = size(features, 2);

loss = 0;

for i = 1:dataSize
    value = WT * features(:,i) * -1;
    if labels(i) == 1
        loss = loss + logSum(0, value);
    else
        loss = loss - value + logSum(0, value);
    end
end

%regularization
loss = loss + lambda/2 * (WT * W - W(1)* W(1));
end


function sum = logSum(x, y)
    sum = 0;
    if x== -inf
        sum = y;
    elseif y== -inf
        sum = x;
    elseif x-y>16
        sum = x;
    elseif x>y
        sum = x + log(1+exp(y-x));
    elseif y-x>16
        sum = y;
    elseif y>x
        sum = y + log(1+exp(x-y));
    end

end


