%%

addpath(genpath('../../source/gpml'));

%%

covfunc = {@covSEiso}; 
hyp.cov = [-5, 0];

likfunc = @likGauss;
hyp.lik = 0;

meanfunc = [];

n = 1000;
x = linspace(-2,2,n)';

K = feval(covfunc{:}, hyp.cov, x);
K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.05, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

covfunc = {@covRQiso}; 
hyp.cov = [-5, 0, -2];

likfunc = @likGauss;
hyp.lik = 0;

meanfunc = [];

n = 1000;
x = linspace(-2,2,n)';

K = feval(covfunc{:}, hyp.cov, x);
K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.05, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

covfunc = {@covNNone}; 
hyp.cov = [-4, 0];

likfunc = @likGauss;
hyp.lik = 0;

meanfunc = [];

n = 1000;
x = linspace(-2,2,n)';

K = feval(covfunc{:}, hyp.cov, x);
K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.05, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,5,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (x(i)^3)/3 + 0.5*x(i)*x(j)*(x(j)-x(i));
        K(i,j) = x(i);
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));
K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.05, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,10,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (x(i)^3)/3 + 0.5*x(i)*x(j)*(x(j)-x(i));
        K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));
K = K +  1e-12*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.2, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,10,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (x(i)^3)/3 + 0.5*x(i)*x(j)*(x(j)-x(i));
        K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));
K = K +  1e-9*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.9, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,1000,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        %K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));

covfunc = {@covPeriodic}; 
hyp.cov = [2, 3, -4];
K2 = feval(covfunc{:}, hyp.cov, x);
K = K .* K2;

K = K +  1e-9*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.12, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,100,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        %K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        K(i,j) = 2 * x(i) + exp(-x(i)) + exp(-x(j)) - exp(x(i) - x(j)) - 1;
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));

K = K +  1e-9*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.12, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,1000,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        %K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        %K(i,j) = 2 * x(i) + exp(-x(i)) + exp(-x(j)) - exp(x(i) - x(j)) - 1;
        K(i,j) = exp(x(i)-x(j));
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));

K = K +  1e-9*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.12, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,10000000,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        %K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        %K(i,j) = 2 * x(i) + exp(-x(i)) + exp(-x(j)) - exp(x(i) - x(j)) - 1;
        %K(i,j) = exp(x(i)-x(j));
        K(i,j) = 1 + x(j) - x(i) + x(i) * x(j) + x(i) ^ 2 * x(j) - ...
                 (1/3)*x(i)^3 + exp(x(i) - x(j)) - exp(-x(i)) - ...
                 exp(-x(j)) - x(i) * exp(-x(j)) - x(j) * exp(-x(j));
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));

K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.45, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;

%%

%covfunc = {@covNNone}; 
%hyp.cov = [-4, 0];

n = 1000;
x = linspace(1,100,n)';
K = zeros(n);
for i = 1:n
    for j = i:n
        %K(i,j) = (1 / 6) * x(i)^2 * (3 * x(j) - x(i));
        %K(i,j) = (1 / 20) * x(i)^3 * (10 * x(j)^2 - 5 * x(i) * x(j) + x(i)^2);
        %K(i,j) = 2 * x(i) + exp(-x(i)) + exp(-x(j)) - exp(x(i) - x(j)) - 1;
        %K(i,j) = exp(x(i)-x(j));
        K(i,j) = exp(-(log(x(i)^2) - log(x(j)^2))^2);
        K(j,i) = K(i,j);
    end
end
%K = ((repmat(x, 1, n).^3)/3) + 0.5 * repmat(x, 1, n) .* repmat(x', n, 1)...
%    .* (repmat(x', n, 1) - repmat(x, 1, n));
%K = triu(K) + triu(K)' - diag(diag(K));

K = K +  1e-6*eye(n)*max(max(K));
y1 = chol(K)' * gpml_randn(0.45, n, 1);

h = figure;
hold on;  
plot(x, y1, 'r', 'Linewidth', 2);
hold off;