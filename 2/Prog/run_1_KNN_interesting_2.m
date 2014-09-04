function pred=run_1_KNN_interesting_2(K, set)

%%%%%%
%KSet = [1,3,5,7,9,11,13,15,17,19];
KSize = size(K,2);
%%%%%%

Data = importdata('USPS-split1.mat');
leaveOneOut = 0;
if strcmp(set,'train')
    testData = Data.X.train;
    leaveOneOut = 1;
elseif strcmp(set,'devel')
    testData = Data.X.devel;
elseif strcmp(set,'test')
    testData = Data.X.test;
else
    testData = [] ;
end

testSize = size(testData,2);
%
pred = zeros(KSize,testSize);
%
featureSize = size(Data.X.train, 1);
trainFeature = Data.X.train;
trainLabel = Data.y.train;
trainDataSize = size(trainFeature,2);
trainNorm = norm(trainFeature(:,1:trainDataSize));
trainFeature = trainFeature - 128 * ones(featureSize,trainDataSize);
for i = 1:testSize
    testVector = testData(:,i) - ones(featureSize,1) * 128;
    if leaveOneOut == 1
        trainFeature = Data.X.train;
        trainFeature(:,i) = [];
        trainLabel = Data.y.train;
        trainLabel(:,i) = [];
        trainDataSize = size(trainFeature,2);
        trainNorm = norm(trainFeature(:,1:trainDataSize));
    end
    
    similarity = testVector' *  trainFeature;
    testNorm = norm(testVector);
    similarity = similarity ./ (testNorm * trainNorm);
    
    error = -1 * similarity;
    
   
    
    [value,index] = sort(error);
    for j = 1:KSize
        k = K(:,j);
        categories = trainLabel(index(1:k));
        result = mode(categories);
        pred(j,i) = result;
    end
end
end



