clear all
warning off
matrices = {'ash958','robot24c1_mat5'};

for i = 1:numel(matrices)
    name = matrices{i};
    fprintf('Testing matrix %s:\n', name);
    snbase = strcat(name,'_');
    load(strcat([char(matrices(i)),'.mat']))
    
    A1 = full(Problem.A);
    nummax = [1,2,4,6,8,10,12,14,16];
    
    cond_plot(nummax,A1,snbase)    
end