clear;
clc;


data_files = {'STARmap_embded'};
for data_file = data_files

    load(['data/' data_file{1} '.mat']);
    output_file = ['results/' data_file{1} '_results.mat'];
    data{1,1} = double(view1);
    data{2,1} = double(view2);
    labels = double(labels)';
    labels = labels+1;
    indc = find(labels==max(labels));
    
    view_nums = size(data,1);
    cluster_nums = max(labels);

    % labels(indc,:)=[];
    best_nmi = 0;
    best_ARI = 0;

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
                        % pred_labels(indc,:)=[];
                        
                        [result_cluster,res] = ClusteringMeasure(labels, pred_labels);
                        nmi(iter) = result_cluster(2);
                        acc(iter) = result_cluster(1);
                        purity(iter) = result_cluster(3);
                        ARI(iter) = result_cluster(4);
                    end
                    
                    if length(unique(res))<cluster_nums
                        continue
                    end
                    if mean(ARI)>best_ARI
                        fprintf('\n.classes=%.2f\n',length(unique(res)));
                        best_ARI = mean(ARI);
                        best_results.nmi = mean(nmi);
                        best_results.nmi_std = std(nmi);
                        best_results.acc = mean(acc);
                        best_results.acc_std = std(acc);
                        best_results.purity = mean(purity);
                        best_results.purity_std = std(purity);
                        best_results.ARI = mean(ARI);
                        best_results.ARI_std = std(ARI);
                        best_results.obj = obj;
                        best_results.option = option;
                        best_results.time = mean(time);
%                         best_results.index = indc;
                        best_results.pred_labels_mapped = res;
                        best_results.pred_labels = pred_labels;
                        fprintf('\n. acc=%.4f, nmi=%.4f, purity=%.4f, ARI=%.4f ...\n', mean(acc),  mean(nmi), mean(purity),mean(ARI));
                        % save(output_file, 'best_results');
                    end
                end
            end
        end
    end

    
end
