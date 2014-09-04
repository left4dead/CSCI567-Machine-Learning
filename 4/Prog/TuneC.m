function [ C ] = TuneC( Data, Label )

indices1 = (Label.train == 1);
indices2 = (Label.train == 2);
class1 = Data.train(indices1,:);
class2 = Data.train(indices2,:);
size1 = size(class1,1);
size2 = size(class2,1);

C = zeros(6,1);
for d=1:3
    regularizer = 0.003;
    
    while regularizer<1
        model = anomalyTrain(class1, d, regularizer);
        pred = anomalyDetect(model, class1);
        ratio = sum(pred==1) / size1;
        if(ratio>0.04 && ratio<0.06)
            C(2*(d-1)+1) = regularizer;
            break;
        end
        regularizer = regularizer + 0.001;
    end
    
    
    regularizer = 0.003;
    
    while regularizer<1
        model = anomalyTrain(class2, d, regularizer);
        pred = anomalyDetect(model, class2);
        ratio = sum(pred==1) / size1;
        if(ratio>0.04 && ratio<0.06)
            C(2*(d-1)+2) = regularizer;
            break;
        end
        regularizer = regularizer + 0.0001;
    end
    
end

end

