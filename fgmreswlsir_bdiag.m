function [opdata,x,iter,fgmresits,flag]= fgmreswlsir_bdiag(A,b,precf,precw,precr,iter_max,gtol)
% fgmreswlsir_bdiag:  FGMRES-based iterative refinement solver for WLS 
% problems in three precisions with block diagonal split preconditioner.
%     opdata = gmresir3(A,b,precf,precw,precr,iter_max,gtol)
%     solves the weighted least-squares problem Ax = b using fgmres-based
%     iterative refinement with at most iter_max ref. steps, and FGMRES 
%     convergence tolerance gtol, with QR factors computed in precision precf:
%       * half if precf = 0,
%       * single if precf = 1,
%       * double if precf = 2,
%     working precision precw:
%       * half if precw = 0
%       * single if precw = 1,
%       * double if precw = 2,
%     and residuals computed at precision precr:
%       * single if precr = 1,
%       * double if precr = 2,
%       * quad if precr = 4
%     Uses split preconditioned FGMRES L1\A/R1 with block split 
%     diagonal preconditioner
%     M = [alpha.*eye(m), Q1*R; R'*Q1', zeros(n)];

if precf ~=0 && precf ~=1 && precf ~= 2, error('precf should be 0, 1 or 2'), end
if precw ~=0 && precw ~=1 && precw ~= 2, error('precw should be 0, 1 or 2'), end
if precr ~=1 && precr ~= 2 && precr ~= 4, error('precr should be 1, 2, or 4'), end

[m,n] = size(A);
q = inf;  % p-norm to use.

% Construct weight matrix D
for j = 1:size(A,1)
	D(j,j) = mp(1/max(abs(A(j,:))),64);
end
D = double(D);

% Construct Schur complement DRAf
DR = chol(D);
DRAf = mp(double(mp(double(DR),64)*mp(double(A),64)),64);

% Compute scalar alpha
alpha = 2^(-1/2)*(min(svd(A)));

if precf == 1
    fprintf('**** Factorization precision is single.\n')
elseif precf == 2
    fprintf('**** Factorization precision is double.\n')
else
    fprintf('**** Factorization precision is half.\n')
end

if precw == 0
    fprintf('**** Working precision is half.\n')
    fp.format = 'h'; chop([],fp);
    A = chop(A);
    b = chop(b);
    [u,~] = float_params(fp.format);
elseif precw == 2
    fprintf('**** Working precision is double.\n')
    A = double(A);
    b = double(b);
    u = eps('double');
else
    fprintf('**** Working precision is single.\n')
    A = single(A);
    b = single(b);
    u = eps('single');
end

if precr == 1
    fprintf('**** Residual precision is single.\n')
elseif precr == 2
    fprintf('**** Residual precision is double.\n')
else
    fprintf('**** Residual precision is quad.\n')
    mp.Digits(34);
end

% Record infinity norm condition number of A
opdata.condA = double(norm(mp(A,64),'inf')*norm(pinv(mp(A,64)),'inf'));

%Compute exact solution and residual via advanpix
xact = (mp(double(DR),64)*(mp(double(A),64)))\((mp(double(DR),64)*mp(double(b),64)));
ract = mp(double(DR),64)*(mp(double(b),64)-mp(double(A),32)*mp(xact,64));
xact = xact*mp(alpha,64);
% Compute norms of exact solution and residual
rtn = norm(ract,2);
xtn = norm(xact,2);

% Compute QR factorization in precf
if precf == 1
    [Q,R] = qr(single(A),0);
    Q = single(Q);
    RR = single(R);                                                         % RR is trapezoidal factor
    [~,DRA]=qr(single(DRAf),0);
elseif precf == 2
    [Q,R] = qr(double(A));
    RR = R;
    [~,DRA]=qr(double(DRAf),0);
else
    fp.format = 'h'; chop([],fp);
    [~,~,~,xmax,~] = float_params(fp.format);
    D1 = diag(1./vecnorm(A));
    mu = 0.1*xmax;
    As = chop(mu*A*D1);
    [Q,R] = house_qr_lp(As,0);                                              % half precision via advanpix
    R = (1/mu)*R*diag(1./diag(D1));
    RR = R(1:n, 1:n);                                                       % RR is trapezoidal factor

    D11 = diag(1./vecnorm(double(DRAf)));
    mu1 = 0.1*xmax;
    As1 = chop(mu1*double(DRAf)*D11);
    [~,R1] = house_qr_lp(As1,0);                                            % half precision via advanpix
    R1 = (1/mu1)*R1*diag(1./diag(D11));
    DRA = R1(1:n, 1:n);                                                     % RR is trapezoidal factor
end
R = RR(1:n, 1:n);                                                           % upper triangular part of RR factor
Q1 = Q(:,1:n);

% Compute and store initial solution and residual in working precision
if precw == 0
    x = chop(R\((Q1'*D*Q1)\(Q1'*D*b)));
    rx = chop(chop(DR*b)-chop(DR*(A*x)));
elseif precw == 2
    x = double(R\((Q1'*D*Q1)\(Q1'*D*b)));
    rx = double(double(DR*b)-double(DR*(A*x)));
else
    x = single(R\((Q1'*D*Q1)\(Q1'*D*b)));
    rx = single(single(DR*b)-single(DR*(A*x)));
end

% Note: when kinf(A) is large, the initial solution x can have 'Inf's in it
% If so, default to using 0 as initial solution
if sum(isinf(single(x)))>0 || sum(isinf(single(rx)))>0
    x =  zeros(size(b,1),1);
    rx = b;
    fprintf('**** Warning: x0 contains Inf. Using 0 vector as initial solution.\n')
end

% Record relative error in computed initial x and r
xerr(1) = norm(double(x)-double(xact),2)./xtn;
rerr(1) = norm(double(rx)-double(ract),2)./rtn;

% Construct (scaled) augmented system
if precw == 0
    Aug_A = chop([alpha.*inv(D), A; A', zeros(n)]);
elseif precw == 2
    Aug_A = double([alpha.*inv(D), A; A', zeros(n)]);
else
    Aug_A = single([alpha.*inv(D), A; A', zeros(n)]);
end

% Record infinity and 2-norm condition numbers of augmented system
opdata.condAugA = cond(Aug_A,'inf');
opdata.condAugA2 = cond(Aug_A,2);

% Construct block diagonal split preconditioners in precw, composed of R
% factors computed in precf
 if precw == 1
    L1 = single([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA']);
    R1 = single([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),1/sqrt(alpha).*DRA]);
elseif precw == 2
    L1 = double([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA']);
    R1 = double([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA]);
 else
    L1 = chop([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(DRA')]);
    R1 = chop([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(DRA)]);
 end

% Form "exact" preconditioned system via Advanpix
extPC = double(L1)\double(Aug_A)/double(R1);

% Record infinity norm condition number of exact preconditioned system
opdata.condMA = cond(double(extPC),'inf');
opdata.condM = cond(double(R1),'inf');

cged = false;
iter = 0; 
fgmresits = [];
fgmreserr = [];

while ~cged
    
    % Increase iteration count; break if hit iter_max
    iter = iter + 1;
    if iter > iter_max, break, end
    
    % Compute residuals in precr
    if precr == 1
        f = single(alpha)*(single(b)-single(rx))-(single(A)*single(x));
        g = single(-A')*single(rx);
    elseif precr == 2
        f = double(alpha)*(double(b)-(double(D)\double(rx)))-(double(A)*double(x));
        g = single(double(-A')*double(rx));
    else
        f = mp(alpha)*(mp(b)-mp(rx))-(mp(A)*mp(x));
        g = mp(mp(-A',32)*mp(rx,32),32);
    end
    
    % Construct right-hand side for augmented scaled system
    Aug_rhs = [f; g];

    nrhs = double(norm(Aug_rhs,inf));

    %Call FGMRES to solve for correction terms 
    if precw == 2
        [d, err, its, flag] = fgmres_dq_eda(Aug_A, zeros(m+n,1), Aug_rhs/nrhs, L1, R1, m+n, 1, gtol);
    else
        [d, err, its, flag] = fgmres_sd_eda(Aug_A, zeros(m+n,1), Aug_rhs/nrhs, L1, R1, m+n, 1, gtol);
    end
       
    d = nrhs*d;
    
    % Pick out updates to x and r from the FGMRES solution
    dr = d(1:m);
    dx = d(m+1:end);
    
    
    % Record the number of iterations fgmres took
    fgmresits = [fgmresits,its];
    
    % Record the final relative (preconditioned) residual norm in FGMRES
    fgmreserr = [fgmreserr,err(end)];
    
    % Record relative (preconditioned) residual norm in each iteration of
    % FGMRES (so we can look at convergence trajectories if need be)
    fgmreserrvec{iter} = err;
    
    % Store previous solution
    xold = x;
    
    % Update solution and residual in precw
    if precw == 0
        rx = chop(chop(rx)+chop(dr));
        x = chop(chop(x)+chop(dx));
    elseif precw == 2
        rx = double(double(rx)+double(dr));
        x = double(double(x)+double(dx));
    else
        rx = single(single(rx)+single(dr));
        x = single(single(x)+single(dx));
    end
       
    % Store relative error in computed x and r
    xerr(iter+1) = mp(norm(mp(x,64)-mp(xact,64),2)./xtn,64);
    rerr(iter+1) = mp(norm(mp(rx,64)-mp(ract,64),2)./rtn,64);
    
    % Check convergence
    if(xerr(iter+1)<= u && rerr(iter+1)<=u)
        x = x/alpha;
        break;
    end
    
    % Compute relative change in solution
    ddx = norm(x-xold,q)/norm(x,q);
    
    % Check if ddx contains infs, nans, or is 0
    if ddx == Inf || isnan(double(ddx))
        break;
    end
    
end

if ((iter >= iter_max) && (xerr(end)>u) && (rerr(end)>u))
    flag = 2; 
end

% Record vector of errors in solution and residual
opdata.xerr = xerr;
opdata.rerr = rerr;

% Record final solution and residual obtained
opdata.x = x;
opdata.x = rx;

% Record information about FGMRES iterations
opdata.fgmresits = fgmresits;
opdata.fgmreserr = fgmreserr;
opdata.fgmreserrvec = fgmreserrvec;


end