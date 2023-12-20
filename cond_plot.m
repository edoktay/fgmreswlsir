function cond_plot(nummax,A1,snbase)
% COND_PLOT
%     constructs left QR and block split diagonal preconditioners for the
%     scaled augmented system for the weighted least squares problems in
%     two precisions.
%     QR factors computed in precision precf:
%       * half if precf = 0,
%       * single if precf = 1,
%       * double if precf = 2,
%     working precision precw:
%       * half if precw = 0
%       * single if precw = 1,
%       * double if precw = 2,
% Outputs: Plots for condition numbers

for ii = 1:numel(nummax)
     DP = diag(logspace(1,nummax(ii),size(A1,1)));
     A = DP\A1;
    kinfA = norm(A,'inf')*norm(pinv(A),'inf');

    for j = 1:size(A1,1)
            D(j,j) = 1/max(abs(A(j,:)));
    end

    dfact = max(max(1,norm(D,2)), norm(inv(D),2));
    s = svd(A);
    alpha = 2^(-1/2)*min(s);
    [m,n] = size(A);
    At = [alpha.*inv(D), A; A', zeros(n)];
    condAt(ii) = cond(At,'inf');
    condD(ii) = cond(D,'inf');
    condA = cond(A,2);
    
    %single Block Diagonal Split
    epss = eps(single(1));
    [Q,R]=qr(single(sqrt(double(D))*double(A)),0);
    SML = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R'/sqrt(alpha)];
    SMR = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R/sqrt(alpha)];
    a = double(SML)\double(At);
    aa = double(a)/double(SMR);
    condSMLs(ii) = cond(double(SML),'inf');
    condPCs(ii) = cond(aa,'inf');
    bounds(ii) = (1+epss*kinfA*dfact)^2;
    
    %half Block Diagonal Split
    fp.format = 'h'; chop([],fp);
    [epsh,~,~,xmax,~] = float_params(fp.format);
    AA = double(sqrt(double(D))*double(A));
    D1 = diag(1./vecnorm(AA));
    mu = 0.1*xmax;
    As = chop(mu*AA*D1);
    [Q,R] = house_qr_lp(As,0); % half precision via advanpix
    R = (1/mu)*R*diag(1./diag(D1));
    R = R(1:n, 1:n);   % RR is trapezoidal factor
    Q1 = Q(:,1:n);
    SML = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R'/sqrt(alpha)];
    SMR = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R/sqrt(alpha)];
     a = double(SML)\double(At);
    aa = double(a)/double(SMR);
    condSMLh(ii) = cond(double(SML),'inf');
    condPCh(ii) = cond(aa,'inf');
    boundh(ii) = (1+epsh*kinfA*dfact)^2;
    
    %double Block Diagonal Split
    [Q,R]=qr(double(double(sqrt(D))*double(A)),0);
    SML = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R'/sqrt(alpha)];
    SMR = [sqrt(alpha.*inv(D)), zeros(m,n); zeros(n,m), R/sqrt(alpha)];
    a = SML\At;
    aa = a/SMR;
    condSMLd(ii) = cond(double(SML),'inf');
    condPCd(ii) = cond(aa,'inf');
    epsd = eps(1);
    boundd(ii) = (1+epsd*kinfA*dfact)^2;

    %single Left QR
    epss = eps(single(1));
    [Q,R]=qr(single(A),0);
    Mqr = [alpha.*inv(D), Q*R; R'*Q', zeros(n)];
    condPCsqr(ii) = cond(double(Mqr)\double(At),'inf');
    
    %half Left QR
    fp.format = 'h'; chop([],fp);
    [epsh,~,~,xmax,~] = float_params(fp.format);
    D1 = diag(1./vecnorm(A));
    mu = 0.1*xmax;
    As = chop(mu*A*D1);
    [Q,R] = house_qr_lp(As,0); % half precision via advanpix
    R = (1/mu)*R*diag(1./diag(D1));
    R = R(1:n, 1:n);   % RR is trapezoidal factor
    Q1 = Q(:,1:n);
    Mqr = [alpha.*inv(D), Q1*R; R'*Q1', zeros(n)];
    condPChqr(ii) = cond(double(Mqr)\double(At),'inf');

    %double Left QR
    [Q,R]=qr(double(A),0);
    Mqr = [alpha.*inv(D), Q*R; R'*Q', zeros(n)];
    condPCdqr(ii) = cond(Mqr\At,'inf');
end

figure
loglog(condD, condAt, 'k--')
hold on;
loglog(condD, condPCh, 'ro-','LineWidth',5);
loglog(condD, condPCs, 'go-','LineWidth',2);
loglog(condD, condPCd, 'bo-');
loglog(condD, (1/epsd).*ones(numel(condD),1), 'k:'); 
loglog(condD, condPChqr,'r+-','LineWidth',5);
loglog(condD, condPCsqr, 'g+-','LineWidth',2);
loglog(condD, condPCdqr, 'b+-');
xlabel('$\kappa_\infty(D)$', 'Interpreter', 'latex');
legend('$\kappa_\infty(\tilde{A})$', '$\kappa_\infty(M_b^{-1/2}\tilde{A}M_b^{-1/2})$, half', ...
    '$\kappa_\infty(M_b^{-1/2}\tilde{A}M_b^{-1/2})$, single', '$\kappa_\infty(M_b^{-1/2}\tilde{A}M_b^{-1/2})$, double', ...
    '$u^{-1}$',  ...
    '$\kappa_\infty(M_l^{-1}\tilde{A})$, half', ...
    '$\kappa_\infty(M_l^{-1}\tilde{A})$, single', '$\kappa_\infty(M_l^{-1}\tilde{A})$, double', ...
    'Interpreter', 'latex', 'Location', 'Northwest');
set(gca,'fontsize',14);
savefig(strcat(snbase,'_syst.fig'))
close all

figure
loglog(condD, condAt, 'k--')
hold on;
loglog(condD, (1/epss).*ones(numel(condD),1), 'k:'); 
loglog(condD, condSMLh, 'rx-','LineWidth',5);
loglog(condD, condSMLs, 'gx-','LineWidth',2);
loglog(condD, condSMLd, 'bx-');
xlabel('$\kappa_\infty(D)$', 'Interpreter', 'latex');
legend('$\kappa_\infty(\tilde{A})$',...
    '$u^{-1}$',...
    '$\kappa_\infty(M_b^{1/2})$, half', ...
    '$\kappa_\infty(M_b^{1/2})$, single', '$\kappa_\infty(M_b^{1/2})$, double', ...
    'Interpreter', 'latex', 'Location', 'Northwest');
set(gca,'fontsize',14);
savefig(strcat(snbase,'_prec.fig'))
close all
end