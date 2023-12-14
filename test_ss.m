% test_ss.m: Tests performance of FGMRES-WLSIR on matrices from SuiteSparse  
% Matrixcollection with (u_f, u, u_r) = (half,single,double) setting.
%
% Note: Requires Multi_precision_NLA_kernels-master and Advanpix 
% multiprecision computing toolbox
%
% Output: test_ss.txt

clear all; close all;

rng(1);
warning off
addpath('Multi_precision_NLA_kernels-master/')
addpath('AdvanpixMCT/')

% Input parameters
maxit = 100;
scale.theta = 0.1; scale.pert = 15;                                         % diagonal perturbation constant
name = {};
a = 0;

% half precision format
fp.format = 'h';                                                             
[u,xmins,xmin,xmax,p,emins,emin,emax] = float_params(fp.format);
chop([],fp);

% Matrices
indlist   = {'photogrammetry','robot24c1_mat5','illc1033','well1033',...
    'Cities','divorce','ash219','ash331','ash608','ash958','WorldCities'};
nn = length(indlist);
eval_ctest = zeros(nn,4);
mn = zeros(nn,1);
ctest = zeros(nn,1);

% Output file
fid1 = fopen('test_ss.txt','w');

for j = 6:10
    fprintf('Processing matrix %d || Total matrices %d\n',j,nn);

    % Load matrix
    load(strcat([char(indlist(j)),'.mat']))
    A = full(Problem.A);
    
    AbsA = abs(A);
    mel(j,1) = max(max(AbsA));
    mel(j,2) = min(AbsA(AbsA>0));
    
    [m,n] = size(A);
    r(1,2) = rank(A);

    % Check if A is rectangular and of full rank
    if ((n == r(1,2)) && (m > n))
        a = a+1;
        
        name = [name,char(indlist(j))];
        act_ind(a,1) = j;
        rows(a,1) = m;
        eval_ctest(a,1) = n;
        eval_ctest(a,4) = m;
        eval_ctest(a,3) = cond(A);
       
        % FGMRES-IR Test
        b = randn(m,1);
       
        % (half,single,double)
        [S_dat1(a),~,its{1,1}(a,1),t_gmres_its{1,1}(a,:),flag{1,1}(a,1)] = fgmreswlsir_qr(A,b,0,1,2,maxit,1e-6);
        [S_dat2(a),~,its{1,2}(a,1),t_gmres_its{1,2}(a,:),flag{1,2}(a,1)] = fgmreswlsir_bdiag(A,b,0,1,2,maxit,1e-6); 

    end
end

% Print matrix properties
fprintf(fid1,'\n\n\nProperties of test matrices \n');
for j=1:a
    jj = act_ind(j,1);
    fprintf(fid1,'%d & %s &(%d,%d)& %6.2e & %6.2e & %6.2e\\\\\n',...
        j,char(name(j)),eval_ctest(j,4),eval_ctest(j,1),eval_ctest(j,3),...
        mel(j,1),mel(j,2));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

%creating a text file to print the GMRES iteration table          
fprintf(fid1,'\n'); fprintf(fid1,'\n');
fprintf(fid1,'half, single, double combination (M_l vs M_b)\n');
for i = 1:a
    f1 = flag{1,1}(i,1);
    f2 = flag{1,2}(i,1);

    t1 = its{1,1}(i,1);
    t2 = its{1,2}(i,1);
    
    t1a  =  t_gmres_its{1,1}(i,:);
    t2a  =  t_gmres_its{1,2}(i,:); 

    fprintf(fid1,'%d & flag = %d & %d &(%s) & flag = %d & %d &(%s)\\\\ \n',...
                i,f1,t1,num2str(t1a),f2,t2,num2str(t2a));
end
fprintf(fid1,'\n'); fprintf(fid1,'\n');

fprintf(fid1,'half, single, double Condition numbers (M_l vs M_b)\n');
for i = 1:a
    t1 = S_dat1(i).condAugA;
    t2 = S_dat1(i).condMA;
    t3 = S_dat2(i).condMA;
    t4 = S_dat2(i).condM;
    fprintf(fid1,'%d & %6.2e & %6.2e & %6.2e & %6.2e\\\\ \n',i,t1,t2,t3,t4);
end

fclose(fid1);
