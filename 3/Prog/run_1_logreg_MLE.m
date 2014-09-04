function [ W_LME ] = run_1_logreg_MLE()

Data = importdata('HW3-USPS-split.mat');

features = Data.X.train;
labels = Data.y.train;
C = Data.C;

DataSize = size(features, 2);
featureSize = size(features,1);
W_LME = zeros(featureSize+1, C);

Cov = zeros(featureSize);

Mean = zeros(featureSize, C);
PI = zeros(1, C);

for i=1:C
    indices = (labels==i);
    featureSubSet = features(:,indices);
    Mean(:,i) = mean(featureSubSet, 2);
    PI(i) = size(featureSubSet,2)/DataSize;
    toMinus = repmat(Mean(:,i), [1, size(featureSubSet,2)]);
    differenceMatrix = featureSubSet - toMinus;
    Cov = Cov + differenceMatrix * differenceMatrix';
end

Cov = Cov ./ DataSize;

for i = 1:C
    W_LME(2:featureSize+1,i) = Cov\Mean(:,i);
    W_LME(1,i) = -0.5 * Mean(:,i)' * W_LME(2:featureSize+1,i) + log(PI(i));
end


end

