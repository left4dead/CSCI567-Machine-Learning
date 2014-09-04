function path = runViterbi()
    load('hw6data.mat');
    totalTime = length(x); % N=8
    S=size(transP,1); % total # of states
    path = zeros(size(x));
    omega = cell(1,totalTime); % trellis
    omega{1} = obsP(:,x(1)).*pi0';
    path(1) = find(omega{1} == max(omega{1}));
    
    for n=2:totalTime
        omega{n} = zeros(S,1);
        % w(z_n=k)
        for k=1:S
            % calculate max w(z_{n-1}) * p(z_n | z_{n-1})
            for kk=1:S
                wPtrans(kk) = omega{n-1}(kk) * transP(kk,k);
            end
            maxTerm = max(wPtrans);
            omega{n}(k) = obsP(k,x(n)) * maxTerm; % w(z_n = k)
        end
        path(n) = find(omega{n} == max(omega{n}));
    end
end