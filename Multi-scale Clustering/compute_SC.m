function [SC] = compute_SC(IDX,data)

view_nums = size(data,1);
temp_sc = 1;
for v = 1:view_nums
   temp_sc = temp_sc * evalclusters(data{1},IDX,'silhouette').CriterionValues;
end
SC = temp_sc^(1/view_nums);