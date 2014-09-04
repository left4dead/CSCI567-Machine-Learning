function [ means ] = newclustering( Data, k )

DataSize = size(Data,2);
indices = randperm(DataSize);
means = Data(:,indices(1:k));
classes = zeros(DataSize,1);
oldDistance = Inf;
newDistance = 0;

while(true)
    newDistance = 0;
    for i = 1:DataSize
        distance = sum( abs(repmat(Data(:,i),1,k) - means) ,1);
        [minDistance, classes(i)] = min(distance);
        newDistance = newDistance + minDistance;
    end
    
    for i = 1:k
        means(:,i) = median(Data(:,classes == i),2);
    end
    
    if(newDistance >= oldDistance * 0.99)
        break;
    end
    oldDistance = newDistance;
    
end

% scatter(Data(1,classes == 1),Data(2,classes == 1),20,'r','fill','o');
% hold on
% scatter(Data(1,classes == 2),Data(2,classes == 2),20,'g','fill','o');
% scatter(Data(1,classes == 3),Data(2,classes == 3),20,'b','fill','o');
% scatter(means(1,1),means(2,1),100,'k','s');
% scatter(means(1,2),means(2,2),100,'k','s');
% scatter(means(1,3),means(2,3),100,'k','s');



end

