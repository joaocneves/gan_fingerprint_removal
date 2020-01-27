


fid = fopen('perf_table_cnn.txt','w');


%% EXPERIMENT 1

exp = 1;
synthetic = {'TPDNE', 'TPDNE', '100F', '100F'};
real = {'VF2', 'CASIA', 'VF2', 'CASIA'};
synthetic_names = {'TPDNE', 'TPDNE', '100K-Face', '100K-Face'};
real_names = {'VGGFace2', 'CASIA-WebFace', 'VGGFace2', 'CASIA-WebFace'};

for i = 1:4
    data = load(fullfile('data',...
        sprintf('real2fake_%s_%s.pretrained.scores',real{i},synthetic{i})));

    scores = data(:,1);
    labels = data(:,2);


    [X,Y,T,auc] = perfcurve(labels,scores,1);
    [eer, eer_i] = roc2eer([X,Y]);
    acc = 1-sum((scores > T(eer_i)) == labels)/numel(labels);
%     acc = sum((scores > 0.5) == labels)/numel(labels);
    
    fprintf(fid, 'A.%d & %s & %s & %s & %s & %.2f & %.2f \\\\ \n', i , ...
        synthetic_names{i},real_names{i},synthetic_names{i},real_names{i}, eer*100, acc*100);
    
end

%% EXPERIMENT 2

exp = 2;
synthetic_train = repmat({'TPDNE', 'TPDNE', '100K-Face', '100K-Face' },1,2);
real_train = repmat({'VGGFace2', 'CASIA-WebFace', 'VGGFace2', 'CASIA-WebFace'},1,2);
synthetic_test = repmat({'100K-Face', '100K-Face', 'TPDNE', 'TPDNE' },1,2);
real_test = { 'VGGFace2', 'CASIA-WebFace', 'VGGFace2', 'CASIA-WebFace', ...
    'CASIA-WebFace', 'VGGFace2', 'CASIA-WebFace', 'VGGFace2' };

data = cell(8,1);

data{1} = load('data\\real2fake_VF2_TPDNE_train_VF2_100F_test.pretrained.scores'); %2.1
data{2} = load('data\\real2fake_CASIA_TPDNE_train_CASIA_100F_test.pretrained.scores'); %2.2
data{3} = load('data\\real2fake_VF2_100F_train_VF2_TPDNE_test.pretrained.scores'); %2.3
data{4} = load('data\\real2fake_CASIA_100F_train_CASIA_TPDNE_test.pretrained.scores'); %2.4

data{5} = load('data\\real2fake_VF2_TPDNE_train_CASIA_100F_test.pretrained.scores'); %2.1
data{6} = load('data\\real2fake_CASIA_TPDNE_train_VF2_100F_test.pretrained.scores'); %2.2
data{7} = load('data\\real2fake_VF2_100F_train_CASIA_TPDNE_test.pretrained.scores'); %2.3
data{8} = load('data\\real2fake_CASIA_100F_train_VF2_TPDNE_test.pretrained.scores'); %2.4

for i = 1:8

    scores = data{i}(:,1);
    labels = data{i}(:,2);


    [X,Y,T,auc] = perfcurve(labels,scores,1);
    [eer, eer_i] = roc2eer([X,Y]); 
%     acc = sum((scores > T(eer_i)) == labels)/numel(labels);
        acc = 1-sum((scores > 0.5) == labels)/numel(labels);

        
    fprintf(fid, 'B.%d & %s & %s & %s & %s & %.2f & %.2f \\\\ \n', i , ...
        synthetic_train{i},real_train{i},synthetic_test{i},real_test{i}, eer*100, acc*100);
    
end

fclose(fid)