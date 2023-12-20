function [condA, condAugA, condMLA, condMBA, condMB]= cond_ml_mb(A,precf,precw)
% COND_ML_MB 
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
% Outputs are condition numbers of:
% condA: input matrix A
% condAugA: augmented system
% condMLA: left preconditioned augmented system
% condMBA: block split diagonal preconditioned augmented system
% condMB: left part of the block split diagonal preconditioner (M_b^{1/2})

if precf ~=0 && precf ~=1 && precf ~= 2, error('precf should be 0, 1 or 2'), end
if precw ~=0 && precw ~=1 && precw ~= 2, error('precw should be 0, 1 or 2'), end

[m,n] = size(A);

for j = 1:size(A,1)
	D(j,j) = mp(1/max(abs(A(j,:))),64);
end
D = double(D);

DR = chol(double(D));
DRAf = mp(double(mp(double(DR),64)*mp(double(A),64)),64);

% Compute scalar alpha
alpha = 2^(-1/2)*(min(svd(A)));

if precw == 0
    fprintf('**** Working precision is half.\n')
    fp.format = 'h'; chop([],fp);
    A = chop(A);
elseif precw == 2
    fprintf('**** Working precision is double.\n')
    A = double(A);
else
    fprintf('**** Working precision is single.\n')
    A = single(A);
end

% Record infinity norm condition number of A
condA = double(norm(mp(A,64),'inf')*norm(pinv(mp(A,64)),'inf'));

% Compute QR factorization in precf
if precf == 1
    fprintf('**** Factorization precision is single.\n')
    [Q,R] = qr(single(A),0);
    Q = single(Q);
    RR = single(R); % RR is trapezoidal factor
    [~,DRA]=qr(single(DRAf),0);
elseif precf == 2
    fprintf('**** Factorization precision is double.\n')
    [Q,R] = qr(double(A));
    RR = R;
    [~,DRA]=qr(double(DRAf),0);
else
    fprintf('**** Factorization precision is half.\n')
    fp.format = 'h'; chop([],fp);
    [~,~,~,xmax,~] = float_params(fp.format);
    D1 = diag(1./vecnorm(A));
    mu = 0.1*xmax;
    As = chop(mu*A*D1);
    [Q,R] = house_qr_lp(As,0); % half precision via advanpix
    R = (1/mu)*R*diag(1./diag(D1));
    RR = R(1:n, 1:n);   % RR is trapezoidal factor
    D11 = diag(1./vecnorm(double(DRAf)));
    mu1 = 0.1*xmax;
    As1 = chop(mu1*double(DRAf)*D11);
    [~,MBR] = house_qr_lp(As1,0); % half precision via advanpix
    MBR = (1/mu1)*MBR*diag(1./diag(D11));
    DRA = MBR(1:n, 1:n);   % RR is trapezoidal factor
end
R = RR(1:n, 1:n);   % upper triangular part of RR factor
Q1 = Q(:,1:n);

% Construct (scaled) augmented system
if precw == 0
    Aug_A = chop([alpha.*inv(D), A; A', zeros(n)]);
elseif precw == 2
    Aug_A = double([alpha.*inv(D), A; A', zeros(n)]);
else
    Aug_A = single([alpha.*inv(D), A; A', zeros(n)]);
end

% Record infinity and 2-norm condition numbers of augmented system
condAugA = cond(Aug_A,'inf');

% Construct left and block diagonal split preconditioners in precw, composed of R
% factors computed in precf
if precw == 1
    ML = single([alpha.*inv(D), single(Q1*R); single(R'*Q1'), zeros(n)]);
    MBL = single([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA']);
    MBR = single([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),1/sqrt(alpha).*DRA]);
elseif precw == 2
    ML = double([alpha.*inv(D), double(Q1*R); double(R'*Q1'), zeros(n)]);
    MBL = double([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA']);
    MBR = double([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*DRA]);
else
    ML = chop([alpha.*inv(D), hgemm(Q1,R); hgemm(R',Q1'), zeros(n)]);
    MBL = chop([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(DRA')]);
    MBR = chop([sqrt(alpha).*inv(DR), zeros(m,n); zeros(n,m),(1/sqrt(alpha)).*single(DRA)]);
end

% Form "exact" left preconditioned system via Advanpix
extPCL = double(ML)\double(Aug_A);

% Record infinity norm condition number of exact left preconditioned system
condMLA = cond(double(extPCL),'inf');

% Form "exact" split preconditioned system via Advanpix
extPCB = double(MBL)\double(Aug_A)/double(MBR);

% Record infinity norm condition number of exact split preconditioned system
condMBA = cond(double(extPCB),'inf');
condMB = cond(double(MBR),'inf');

end