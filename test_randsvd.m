clear all; close all;
rng(1);
warning off

addpath('Multi_precision_NLA_kernels-master/')
addpath('AdvanpixMCT/')

% Input parameters
maxit = 100;
scale.theta = 0.1; scale.pert = 15; % diagonal perturbation constant
fp.format = 'h'; % low precision format to be considered
rng(1)
A1 = gallery('randsvd',[100,10],1e2,3);

nummax = [1,2,4,6,8,10,12,14,16];
nn = numel(nummax);
eval_ctest = zeros(nn,4);
mn = zeros(nn,1);
ctest = zeros(nn,1);

fid1 = fopen('test_randsvd.txt','w');
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
chop([],fp);

name = {};
a = 0;
for j=1:nn
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);
    DP = diag(logspace(1,nummax(j),100));
    A = DP*A1;
    
    AbsA = abs(A);
    mel(j,1) = max(max(AbsA));
    mel(j,2) = min(AbsA(AbsA>0));
    
    
    [d,r(1,1)] = size(A);
    r(1,2) = rank(A);

    if ((r(1,1) == r(1,2)) && (d > r(1,1)))
        a = a+1;
        
        act_ind(a,1) = j;
        rows(a,1) = d;
        eval_ctest(a,1) = r(1,1);
        eval_ctest(a,4) = d;
        eval_ctest(a,3) = cond(A);

        % (half,single)
        [condA(a), condAugA(a), condMLA(a), condMBA(a), condMB(a)]= cond_ml_mb(A,0,1);

    end
end

fprintf(fid1,'Matrix No. & size(A)  & max(max(|A|)) & min(|A|(|A|>0))\n');
for j = 1:a
    jj = act_ind(j,1);
    fprintf(fid1,'%5d      & (%d,%d) &    %6.2e   & %6.2e\n',j,eval_ctest(j,4),eval_ctest(j,1),mel(j,1),mel(j,2));             
end      
fprintf(fid1,'\n'); fprintf(fid1,'\n');

fprintf(fid1,'Matrix No. &  condA   & condAugA & condMLA  & condMBA  & condMB \n');
for i = 1:a
    t0 = condA(i);
    t1 = condAugA(i);
    t2 = condMLA(i);
    t3 = condMBA(i);
    t4 = condMB(i);
    fprintf(fid1,'%5d      & %6.2e & %6.2e & %6.2e & %6.2e & %6.2e \n',i,t0, t1,t2,t3,t4);
end

fclose(fid1);
