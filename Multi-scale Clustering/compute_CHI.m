function [CHI] = compute_CHI(IDX,data)

view_nums = size(data,1);
temp_CHI = 1;
for v = 1:view_nums
   temp_CHI = temp_CHI * evalclusters(data{1},IDX,'CalinskiHarabasz').CriterionValues;
end
CHI = temp_CHI^(1/view_nums);