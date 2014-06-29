
a='Load the data, it should contain X and y.'
load 'tmpWqk4yD.mat'

%Load GPML
addpath(genpath('/Users/JamesLloyd/Documents/MATLAB/GPML/gpml-matlab-v3.1-2010-09-27'));

%Set up model.
meanfunc = {@meanConst}
hyp.mean = mean(y)

covfunc = {@covMask, {[0 0 0 1], @covSEiso}}
hyp.cov = [ 0.0 0.0 ]

likfunc = @likGauss
hyp.lik = -2.30080756849

[hyp_opt, nlls] = minimize(hyp, @gp, -300, @infExact, meanfunc, covfunc, likfunc, X, y);
best_nll = nlls(end)

%Compute Hessian numerically for laplace approx
num_hypers = length(hyp.cov);
hessian = NaN(num_hypers, num_hypers);
delta = 1e-6;
a='Get original gradients';
[nll_orig, dnll_orig] = gp(hyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
for d = 1:num_hypers
    dhyp_opt = hyp_opt;
    dhyp_opt.cov(d) = dhyp_opt.cov(d) + delta;
    [nll_delta, dnll_delta] = gp(dhyp_opt, @infExact, meanfunc, covfunc, likfunc, X, y);
    hessian(d, :) = (dnll_delta.cov - dnll_orig.cov) ./ delta;
end
hessian = 0.5 * (hessian + hessian');

save( '/Users/JamesLloyd/temp/tmppdl65_.out', 'hyp_opt', 'best_nll', 'nlls', 'hessian' );
%exit();

fprintf('\nWriting completion flag\n');
ID = fopen('/Users/JamesLloyd/temp/tmpCv8yxX.flg', 'w');
fprintf(ID, 'Goodbye, world');
fclose(ID);
fprintf('\nGoodbye, World\n');
quit()
