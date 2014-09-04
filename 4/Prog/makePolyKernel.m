function [K] = makePolyKernel(x1, x2, c , d)

K = x1 * x2' + c;
K = K .^ d;

end