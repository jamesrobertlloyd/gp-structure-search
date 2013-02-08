x = linspace(0, 10, 50)';
y = 0 * x + sin(x*3);
y = y + randn(size(y))*0.2;
plot (x, y, '+');

scale = 0;
x = x * exp(scale);
y = y * exp(scale);

covfunc = {@covSEiso};
hyp.cov = [scale, scale];

likfunc = @likGauss;
hyp.lik = std(y) / 10;

meanfunc = {@meanZero};
hyp.mean = [];

[hyp_opt, nlls] = minimize(hyp, @gp, -100, @infExact, ...
                   meanfunc, covfunc, likfunc, x, y);

cov_hyp = hyp_opt.cov
mean_hyp = hyp_opt.mean
lik_hyp = hyp_opt.lik

nll = nlls(end)


scale = 5;
x = x * exp(scale);
y = y * exp(scale);

covfunc = {@covSEiso};
hyp.cov = [scale, scale];

likfunc = @likGauss;
hyp.lik = std(y) / 10;

meanfunc = {@meanZero};
hyp.mean = [];

[hyp_opt, nlls] = minimize(hyp, @gp, -100, @infExact, ...
                   meanfunc, covfunc, likfunc, x, y);

cov_hyp2 = hyp_opt.cov;
mean_hyp2 = hyp_opt.mean;
lik_hyp2 = hyp_opt.lik;

cov_hyp2 - cov_hyp
lik_hyp2 - lik_hyp

nll2 = nlls(end);
nll - nll2