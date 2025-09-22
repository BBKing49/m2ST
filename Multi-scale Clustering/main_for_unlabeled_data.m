clear;
clc;


data_files = {'mouse_hippocampus_embded'};
for data_file = data_files

    load(['data/' data_file{1} '.mat']);
    output_file = ['results/' data_file{1} '_results.mat'];

    data{1,1} = double(view1);
    data{2,1} = double(view2);
    labels = [];

    view_nums = size(data,1);
    cluster_nums = 10;
%     best_nmi = 0;
    best_SC = 0;
    
    for lambda1 = 10.^(-5:2)
        for lambda2 = 10.^(-5:-1)
            for lambda3 = 10.^(-5:2)
                for dim = [10:10:100]
                    option.lambda1 = lambda1;
                    option.lambda2 = lambda2;
                    option.lambda3 = lambda3;
                    for v = 1:view_nums
                        %                             sdim{v,1} = dim;
                        if dim>size(data{v},2)
                            sdim{v,1} = size(data{v},2);%floor(size(data{v},2)/10)*10;
                        else
                            sdim{v,1} = dim;
                        end
                    end
                    option.sdim = sdim;
                    option.cdim = cluster_nums;
                    option.numClust = cluster_nums;
                    option.Maxitems = 100;
                    for iter = 1:1
                        tic;
                        [U,obj] = OMC_DR(data,option,iter,labels);
                        time(iter) = toc;
                        pred_labels = vec2lab(U');
                        DBI(iter) = compute_dbi(pred_labels, data);
                        SC(iter) = compute_SC(pred_labels, data);
                        CHI(iter) = compute_CHI(pred_labels, data);
                    end
                    if mean(SC)>best_SC
                        best_SC = mean(SC);
                        best_results.SC = mean(SC);
                        best_results.SC_std = std(SC);
                        best_results.DBI = mean(DBI);
                        best_results.DBI_std = std(DBI);
                        best_results.CHI = mean(CHI);
                        best_results.CHI_std = std(CHI);

                        best_results.obj = obj;
                        best_results.option = option;
                        best_results.time = mean(time);
                        best_results.pred_labels = pred_labels;
                        fprintf('\n. SC=%.4f, DBI=%.4f, CHI=%.4f ...\n', mean(SC),  mean(DBI), mean(CHI));
                        % save(output_file, 'best_results');
                    end
                end
            end
        end
    end

    % save(output_file, 'best_results');
end
