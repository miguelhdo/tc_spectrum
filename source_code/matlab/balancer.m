clear all
close all
%gpuDevice(1)
dir = "../../dataset/waveforms/";
%filename = "waveforms_2G_n_multi_mobile_app_2_only_qos";
filename = "waveforms_24042020_2G_n_6_mobile_app";
waveform_to_waveform = dir+filename+".mat";
load(waveform_to_waveform)
num_labels = numel(Y{1});
label_selected = 4;
Y_array=(reshape(cell2mat(Y),num_labels, size(Y,2)));
Y_array = Y_array(label_selected,:);
%Y_2 = reshape(Y_array, 3, size(Y,2));
%catnames = {'mgmt','ctr','qos'};
%OriginalYTrain = discretize(Y_array,[0 1 2 3],'categorical',catnames);
%OriginalYTrain=OriginalYTrain';

summaryClasses = tabulate(Y_array);
summaryClasses
pause
sizeMinClass = min(summaryClasses(:,2));
classes_labels = summaryClasses(:,1);
num_classes = numel(classes_labels);

indexes_balanced_classes = zeros(1,sizeMinClass*num_classes);
ind_0 = 1;
ind_1 = sizeMinClass;
for i = 1:numel(classes_labels)
    temp = find(Y_array==classes_labels(i));
    idx = randperm(size(temp,2),sizeMinClass);
    temp = temp(idx);
    indexes_balanced_classes(ind_0:ind_1)=temp;
    ind_0=ind_1+1;
    ind_1=ind_0+sizeMinClass-1;
end

X=X(indexes_balanced_classes);
Y=Y(indexes_balanced_classes);
dir_out = "../../dataset/waveforms/";
waveforms_file = dir_out+"waveforms_"+filename+"_balanced.mat";
save(waveforms_file,'X','Y', '-v7.3')