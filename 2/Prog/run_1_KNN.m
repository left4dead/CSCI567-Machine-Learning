function pred=run_1_KNN(K, p,set)

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
for i = 1:testSize
    testVector = testData(:,i);
    if leaveOneOut == 1
        trainFeature = Data.X.train;
        trainFeature(:,i) = [];
        trainLabel = Data.y.train;
        trainLabel(:,i) = [];
        trainDataSize = size(trainFeature,2);
    end
    
    temp = repmat(testVector,1, trainDataSize);
    temp = abs(temp - trainFeature);
    if isinf(p) == 0
        temp = temp.^p;
        error = ones(1,featureSize) * temp;
    else
        error = max(temp);
    end
    
    [value,index] = sort(error);
    for j = 1:KSize
        k = K(:,j);
        categories = trainLabel(index(1:k));
        result = mode(categories);
        pred(j,i) = result;
    end
end
end



