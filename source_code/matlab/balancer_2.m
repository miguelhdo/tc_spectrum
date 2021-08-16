clear all
close all
%gpuDevice(1)
dir = "../../dataset/waveforms/";
filename = "waveforms_07082020_2G_n_mobile_L8";
%filename = "waveforms_16042020_2G_n_unknown_unknown_v2WLAN_CLASS";
waveform_to_waveform = dir+filename+".mat";
load(waveform_to_waveform)

label_selected = 4;
num_labels = numel(Y{1});

Y_array=(reshape(cell2mat(Y),num_labels, size(Y,2)));
Y_array = Y_array(label_selected,:);
if label_selected==3
    task = "_app-type";
elseif label_selected==1
    task = "_wlan-frame";
elseif label_selected==4
    task = "_app";
end

summaryClasses = tabulate(Y_array);

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
X_payload=X_payload(indexes_balanced_classes);
Y_array=(reshape(cell2mat(Y),num_labels, size(Y,2)));
Y_array = Y_array(label_selected,:);
summaryClasses_2 = tabulate(Y_array);

dir_out = "../../dataset/waveforms/";
waveforms_file = dir_out+filename+task+"_balanced.mat";
save(waveforms_file,'X','X_payload', 'Y', '-v7.3')