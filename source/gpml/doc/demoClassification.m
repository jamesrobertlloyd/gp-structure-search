disp(' '); disp('clear all, close all');
clear all, close all
disp(' ')

disp('n1 = 80; n2 = 40;                   % number of data points from each class');
n1 = 80; n2 = 40;
disp('S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices');
S1 = eye(2); S2 = [1 0.95; 0.95 1];
disp('m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means');
m1 = [0.75; 0]; m2 = [-0.75; 0];
disp(' ')

disp('x1 = bsxfun(@plus, chol(S1)''*gpml_randn(0.2, 2, n1), m1);')
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
disp('x2 = bsxfun(@plus, chol(S2)''*gpml_randn(0.3, 2, n2), m2);')         
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         
disp(' ');

disp('x = [x1 x2]''; y = [-ones(1,n1) ones(1,n2)]'';');
x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
disp('plot(x1(1,:), x1(2,:), ''b+''); hold on;');
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on;
disp('plot(x2(1,:), x2(2,:), ''r+'');');
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);
disp(' ');

disp('[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);');
[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
disp('t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs');
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
disp('tmm = bsxfun(@minus, t, m1'');');
tmm = bsxfun(@minus, t, m1');
disp('p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));');
p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
disp('tmm = bsxfun(@minus, t, m2'');');
tmm = bsxfun(@minus, t, m2');
disp('p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));');
p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
set(gca, 'FontSize', 24);
disp('contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9]);');
contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9]);
[c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
%print -depsc f5.eps
disp(' '); disp('Hit any key to continue...'); pause

disp(' ');
disp('meanfunc = @meanConst; hyp.mean = 0;');
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);');
covfunc = @covSEard;   hyp.cov = log([1 1 1]);
disp('likfunc = @likErf;');
likfunc = @likErf;
disp(' ');

disp('hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);');
hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
disp('[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));');
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
disp(' ');
figure
set(gca, 'FontSize', 24);
disp('plot(x1(1,:), x1(2,:), ''b+''); hold on;');
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on;
disp('plot(x2(1,:), x2(2,:), ''r+'')');
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
disp('contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);');
contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
%print -depsc f6.eps
disp(' ');
