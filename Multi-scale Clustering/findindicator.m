function outVec = findindicator(X,V,view_weight,i)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
clu_nums=size(V{1},2);
obj = zeros(clu_nums, 1);
tmp = eye(clu_nums);
view_nums = size(V,1);
for v = 1:view_nums
    for c=1:clu_nums
        obj(c,1) = obj(c,1) + view_weight{v}*(norm(X{v}(:,i) - V{v}(:,c)).^2);
    end
end
[min_val, min_idx] = min(obj);
outVec = tmp(:, min_idx);
end


