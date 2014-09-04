function [pred] = run_2_LR()
    load('wine.mat');
    lambda = 1;
    [N D] = size(X.train);
    linearTrainInput = [X.train];
    linearTrainOutput = [y.train];
    
    % "Linear + Quadratic" features
    for i = 1:N
       row_i =  linearTrainInput(i,:);
       for j = 1:D
           for k = j:D
               row_i = [row_i row_i(j)*row_i(k)];
           end
       end
       linearQuadTrainInput(i,:) = row_i;
    end
    
    % "Linear + Quadratic + Cubic" features
    kIndex = D+1;
    jumpFactor = D;i
    linearQuadCubeTrainInput = [];
    for i = 1:N
       row_i =  linearQuadTrainInput(i,:);
       for j = 1:D
           for k = kIndex:size(linearQuadTrainInput,2)
               row_i = [row_i row_i(j)*row_i(k)];
           end
           kIndex = kIndex + jumpFactor;
           jumpFactor = jumpFactor - 1;
       end
       linearQuadCubeTrainInput(i,:) = row_i;
       kIndex = D+1;
       jumpFactor = D;
    end
    
    linearTrainInput = [ones(N,1) , linearTrainInput];
    linearQuadTrainInput = [ones(N,1) , linearQuadTrainInput];
    linearQuadCubeTrainInput = [ones(N,1) , linearQuadCubeTrainInput];
    
    % w^{LMS} for Linear + Quadratic + Cubic features
    E = eye(size(linearQuadCubeTrainInput,2));
    E(1,1) = 0;
    wLinearQuadCubeFeature = (linearQuadCubeTrainInput'*linearQuadCubeTrainInput + lambda*E) \ (linearQuadCubeTrainInput'*linearTrainOutput);
    
    % test set features
    [P Q] = size(X.test);
    linearTestInput = [X.test];
    
    % "Linear + Quadratic" features
    for i = 1:P
       row_i =  linearTestInput(i,:);
       for j = 1:Q
           for k = j:Q
               row_i = [row_i row_i(j)*row_i(k)];
           end
       end
       linearQuadTestInput(i,:) = row_i;
    end
    
    % "Linear + Quadratic + Cubic" features
    kIndex = Q+1;
    jumpFactor = Q;
    linearQuadCubeTestInput = [];
    for i = 1:P
       row_i =  linearQuadTestInput(i,:);
       for j = 1:Q
           for k = kIndex:size(linearQuadTestInput,2)
               row_i = [row_i row_i(j)*row_i(k)];
           end
           kIndex = kIndex + jumpFactor;
           jumpFactor = jumpFactor - 1;
       end
       linearQuadCubeTestInput(i,:) = row_i;
       kIndex = D+1;
       jumpFactor = D;
    end
    
    linearTestInput = [ones(P,1) , linearTestInput];
    linearQuadTestInput = [ones(P,1) , linearQuadTestInput];
    linearQuadCubeTestInput = [ones(P,1) , linearQuadCubeTestInput];
    
    % prediction
    pred = linearQuadCubeTestInput * wLinearQuadCubeFeature;