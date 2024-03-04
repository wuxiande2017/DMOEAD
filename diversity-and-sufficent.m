clear
load('F:\data\PaviaC\Pavia_gt.mat')
temp = pavia_gt(1:600,1:600);
index_map = temp;
num_there = 5; %确定补全数据
upbound_num = 30; %补全数目的上限
q = 1;
spatial_half = 59;
j=spatial_half;
i=spatial_half;
spatial_x = 600;
spatial_y = 600;
for i=60:120:spatial_x-spatial_half
         j=spatial_half+1;
     while j>spatial_half && j<=spatial_y-spatial_half
        if index_map(i,j)~=0 && index_map(i,j)~=20 && rem(i,3)==0 && rem(j,3)==0
            loc(q,1) = i;
            loc(q,2) = j;
%             label_index(q)=index_map(i,j);
            q = q+1;
            index_map((i-spatial_half):(i+spatial_half),(j-spatial_half):(j+spatial_half))=20;
%             imagesc(index_map);
%             pause(0.1)
            j=j+120;
        else
            j=j+1;
        end    
     end
end
imagesc(index_map)
for i=1:length(loc)
    label(i) =temp(loc(i,1),loc(i,2));
end

for i=1:9
    leng(i)=length(find(label==i));
end
% most_class = find(leng>=num_there);
most_class = [2,4,6,7,8,9];
i=1;
for i =1:length(label)
    if label(i)==most_class(1) || label(i)==most_class(2) || label(i)==most_class(3)|| label(i)==most_class(5) || label(i)==most_class(6)
    else
        label(i)=0;
        loc(i,1:2)=0; 
    end
end

[class_loc_x, class_loc_y]= find(temp==4);
index = randperm(length(class_loc_x),1);
x_selec = class_loc_x(index);
y_selec = class_loc_y(index);
label = [label,4];
loc = [loc;[x_selec,y_selec]];

label(find(label==0))=[];
loc(find(loc==0))=[];
loc = reshape(loc,[length(label),2]);

most_class(find(most_class==7))=[];
for i=1:length(most_class)
% for i=1:2
    class_index = most_class(i);
    base_num = length(find(label==class_index));
    [class_loc_x, class_loc_y]= find(temp==class_index);
    recurrent_time = 100000;
    recurrent_time_count=0;
    distance_there = 1000; %确定随机生成样本是否接受
    while base_num < upbound_num
        index = randperm(length(class_loc_x),1);
        x_selec = class_loc_x(index);
        y_selec = class_loc_y(index);
        if  rem(x_selec,3)==0 && rem(y_selec,3)==0 
        %计算距离
            candidate_cro = loc(find(label==class_index),1:2);
            distance = abs(candidate_cro - [repmat(x_selec,length(candidate_cro),1),repmat(y_selec,length(candidate_cro),1)]);
            if min(sum(distance,2))>distance_there && x_selec>spatial_half && x_selec<=spatial_x-spatial_half && y_selec>spatial_half && y_selec<=spatial_y-spatial_half
                loc = [loc;[x_selec,y_selec]];
                label = [label,class_index];
                base_num = base_num +1;
            else
            %如果找100次找不到就缩小阈值
                recurrent_time_count = recurrent_time_count +1;
                if recurrent_time_count == recurrent_time
                    distance_there = distance_there - 20;
                    if distance_there <0 
                        error('阈值小于0')
                    end
                    recurrent_time_count = 0;
                end
            end
        end
    end
end

save('F:\wxd_第四个工作\multitask_Pavia\loc_pan.mat','label','loc')
%检查上面类别和下面的类别


for i=1:length(loc)
    aaa = temp;
    aaa((loc(i,1)-spatial_half):(loc(i,1)+spatial_half),(loc(i,2)-spatial_half):(loc(i,2)+spatial_half))=20;
    imagesc(aaa)
end



