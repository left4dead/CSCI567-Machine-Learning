function [ result ] = experiment ( Data, Label, C )


supportNumber = zeros(6,1);
r = zeros(6,1);
falseAcceptRate = zeros(6,1);
falseRejectRate = zeros(6,1);


for d=1:3
    
    for class=1:2
        indices = (Label.train == class);
        data = Data.train(indices,:);
        model = anomalyTrain(data, d, C(d*2 -2 + class));
        supportNumber(d*2 -2 + class) = size(model.supportVector,1);
        r(d*2 -2 + class) = model.r;
        pred = anomalyDetect(model, Data.test);
        AcceptCount = sum(pred == 0);
        RejectCount = sum(pred == 1);
        FalseAccept = sum(pred ==0 & Label.test~=class);
        FalseReject = sum(pred ==1 & Label.test==class);
        falseAcceptRate(d*2 -2 + class) = FalseAccept / AcceptCount;
        falseRejectRate(d*2 -2 + class) = FalseReject / RejectCount;
    end
    
end

result = struct('supportNumber', supportNumber, 'r', r, 'falseAcceptRate', falseAcceptRate , 'falseRejectRate', falseRejectRate);

end

