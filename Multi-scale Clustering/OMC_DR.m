function [U,obj]=OMC_DR(data,option,iter,labels)

view_nums = size(data,1);

lambda1 = option.lambda1;
lambda2 = option.lambda2;
lambda3 = option.lambda3;
spec_dim = option.sdim;
com_dim = option.cdim;
numClust = option.numClust;
Maxitems = option.Maxitems;

% inital
sum1=0;
sum2=0;
for v = 1:view_nums
%     data{v} = data{v}';
    data{v} = mapminmax(data{v}',0,1)';
%     data{v} = Normalization(data{v});
    view_weight{v,1} = 1/(view_nums+1);
    [N,d] = size(data{v});

    S{v,1} = get_inital(N,spec_dim{v},1000+iter*0.1);%1000  +iter*0.035 1000DLPFC 100 others
    S{v,1} = S{v,1}';
%         rand('seed',v*100);
%     S{v,1} = rand(N,spec_dim{v});
    P{v,1} = rand(spec_dim{v},d);    
    W{v,1} = rand(com_dim,d); 
    X{v,1} = S{v,1};
    V{v,1} = rand(spec_dim{v},numClust);
end

% 第v+1个为共性视角
rand('seed',100+iter*0.1);
H = rand(N,com_dim)';

X{view_nums+1,1} = H;
view_weight{view_nums+1,1} = 1/(view_nums+1);
V{view_nums+1,1} = rand(com_dim,numClust);
g0 = ceil(rand(1,N)*numClust);
U = ToM(g0',numClust,N);
obj = zeros(Maxitems,1);

for iter = 1:Maxitems
%     obj(iter) = get_obj(data,H,S,W,P,U,V,view_weight,option);
%     if iter>1 && abs(obj(iter)-obj(iter-1))<1e-6
%         break;
%     end
%     obj = Normalization(obj);
    sum1 = zeros(com_dim);
    sum2 = zeros(com_dim,N);
    sum3 = zeros(com_dim,com_dim);
    sum_weight = 0;
   
    for v = 1:view_nums+1
        sum_var = norm(X{v}-V{v}*U,2).^2;
        sum_var = exp(-lambda2*sum_var);
        sum_weight = sum_weight+sum_var;
    end
    for v = 1:view_nums
        
        % update S view weight
        sum_var = norm(X{v}-V{v}*U,2).^2;
        acc_vw = exp(-lambda2*sum_var)/sum_weight;
        view_weight{v} = acc_vw;
        
        % update S
        temp_S = ((lambda1+acc_vw)*eye(spec_dim{v})+P{v}*P{v}')\(P{v}*data{v}'-P{v}*W{v}'*H+acc_vw*V{v}*U);
        temp_S = Normalization(temp_S);
        S{v} = temp_S;
        X{v} = S{v};
        
        % update P
%         temp_P = pinv(temp_S'*temp_S)*(temp_S'*data{v}-temp_S'*H*W{v});
        temp_P = (S{v}*S{v}')\(S{v}*data{v}-S{v}*H'*W{v});
        temp_P = Normalization(temp_P);
        P{v} = temp_P;
        
        % update W
        temp_W = (H*H')\(H*data{v}-H*S{v}'*P{v});
        temp_W = Normalization(temp_W);
        W{v} = temp_W;
        

        % update S V
%         temp_V = (S{v}*U')*pinv(U*U');
        temp_V = (view_weight{v}*temp_S*U')*pinv(view_weight{v}*U*U'+lambda3*V{v}'*V{v}-lambda3*eye(numClust));
        V{v} = temp_V;


        sum1 = sum1 + W{v}*W{v}';
        sum2 = sum2 + (W{v}*data{v}'-W{v}*P{v}'*S{v});
    end
    
    %update H view_weight
    sum_var = norm(X{view_nums+1}-V{view_nums+1}*U,2).^2;
    acc_vw = exp(-lambda2*sum_var)/sum_weight;
    view_weight{view_nums+1} = acc_vw;
    
    % update H
%     temp_H = pinv((1+acc_vw)*eye(N)+sum1+lambda2*D)*(sum2+U'*V{view_nums+1}');
    temp_H = ((lambda1+acc_vw)*eye(com_dim)+sum1)\(sum2+acc_vw*V{view_nums+1}*U);

    temp_H = Normalization(temp_H);
    H = temp_H;
    X{view_nums+1} = H;
    
    % update H V
%     temp_V = (H*U')*pinv(U*U');
    temp_V = (view_weight{view_nums+1} *H*U')*pinv(view_weight{view_nums+1}*U*U'+lambda3*V{view_nums+1}'*V{view_nums+1}-lambda3*eye(numClust));
    V{view_nums+1} = temp_V;


    % update U
    parfor i = 1:N
        U(:,i) = findindicator(X, V,view_weight,i);
    end

   
%     if mod(iter, 1) == 0
%         X_iter = X;
%         U_iter = U;
%         V_iter = V;
%         pred_labels = vec2lab(U');
%         for v = 1:view_nums+1
%             X_iter{v} = X_iter{v}';
%             V_iter{v} = V_iter{v}';
%         end
%         save(['T-SNE/res_' int2str(iter) '_der.mat'],'X_iter','U_iter','V_iter','pred_labels','labels');
%     end
end
end

function [linshi_U] = get_inital(new_dim,dim,r)
    rand('seed',r);
    linshi_U = rand(new_dim,dim);
    if new_dim > dim
        X = orth(linshi_U);
    else
        X = (orth(linshi_U'))';
    end
end

function [D] = get_D(X)
    N = size(X,1);
    D = zeros(N);
    for j = 1:N
        D(j,j) = 1/norm(X(j,:));
    end
end

function [X] = Normalization(X)

    X(isnan(X)) = 0;
    X(isinf(X)) = 1e5;
    norm_mat = repmat(sqrt(sum(X.*X,2)),1,size(X,2));
    for i = 1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    X = X./norm_mat;

end


function [final_obj] = get_obj(X,H,S,W,P,U,V,view_weight,option)

    lambda1 = option.lambda1;
    lambda2 = option.lambda2;
    lambda3 = option.lambda3;
    view_nums = size(X,1);
    obj = zeros(view_nums+1,1);

    for v = 1:view_nums
        S{v} = Normalization(S{v});
        S_clu = view_weight{v}*norm(Normalization(S{v}-V{v}*U),2) + lambda2*view_weight{v}*log(view_weight{v});
        obj(v) =  norm(Normalization(X{v}-H'*W{v}-S{v}'*P{v}))  + lambda1*norm(S{v}', 2) + S_clu + lambda3*norm(Normalization(V{v}'*V{v})-eye(option.numClust),2);
    end
    H = Normalization(H);
    H_clu = view_weight{view_nums+1}*norm(Normalization(H-V{view_nums+1}*U),2) + lambda2*view_weight{view_nums+1}*log(view_weight{view_nums+1});
    obj(view_nums+1) =  H_clu + lambda1*norm(H', 2)+lambda3*norm(Normalization(V{view_nums+1}'*V{view_nums+1}-eye(option.numClust)),2);
    
    final_obj = sum(obj);
end
