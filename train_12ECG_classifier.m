function  model = train_12ECG_classifier(input_directory,output_directory)

disp('Loading data...')

% Find files.
input_files = {};
for f = dir(input_directory)'
    if exist(fullfile(input_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'mat')
        input_files{end + 1} = f.name;
    end
end

% read number of unique classes
%classes = get_classes(input_directory,input_files);
classes=["270492004" "164889003" "164890007" "426627000" "713427006" "713426002" "445118002" "39732003" "164909002" "251146004" "698252002" "10370003" "284470004" "427172004" "164947007" "111975006" "164917005" "47665007" "59118001" "427393009" "426177001" "426783006" "427084000" "63593006" "164934002" "59931005" "17338001"];
num_classes = length(classes);
num_files = length(input_files);
Total_data=cell(1,num_files);
Total_header=cell(1,num_files);


% Iterate over files.
for i = 1:num_files
    disp(['    ', num2str(i), '/', num2str(num_files), '...'])
    
    % Load data.
    file_tmp=strsplit(input_files{i},'.');
    tmp_input_file = fullfile(input_directory, file_tmp{1});
    
    [data,hea_data] = load_challenge_data(tmp_input_file);
    
    Total_data{i}=data;
    Total_header{i}=hea_data;
    
end

disp('Training model..')

label=zeros(num_files,num_classes);

for i = 1:num_files
    
    disp(['    ', num2str(i), '/', num2str(num_files), '...']);
    
    data = Total_data{i};
    header_data = Total_header{i};
    
    tmp_features = get_12ECG_features(data,header_data);
    
    features(i,:)=tmp_features;

    for j = 1 : length(header_data)
        if startsWith(header_data{j},'#Dx')
            tmp = strsplit(header_data{j},': ');
            tmp_c = strsplit(tmp{2},',');
            for k=1:length(tmp_c)
                if tmp_c{k}=="164884008"
                    label(i,27)=1;
                end
                idx=find(strcmp(classes,tmp_c{k}));
                if idx==0
                    label(i,:)=0;
                else
                label(i,idx)=1;
                end
            end
            break
        end
    end  
end
% model = mnrfit(features,label,'model','hierarchical');

t = ClassificationTree.template('minleaf',1);
%AF MODEL
afdata=features((label(:,2)==1),:);
afrestdata=features((label(:,2)==0),:);
afdataf=[afdata; afrestdata];
aflabels=[ones(1,size(afdata,1)),zeros(1,size(afrestdata,1))]';
modelaf = fitensemble(afdataf,aflabels,'RusBoost',30,t,'type','classification');
modelaf=compact(modelaf);

% AFL MODEL
afldata=features((label(:,3)==1),:);
aflrestdata=features((label(:,3)==0),:);
afldataf=[afldata; aflrestdata];
afllabels=[ones(1,size(afldata,1)),zeros(1,size(aflrestdata,1))]';
modelafl = fitensemble(afldataf,afllabels,'RusBoost',30,t,'type','classification');
modelafl=compact(modelafl);

% BRADY MODEL
bradydata=features((label(:,4)==1),:);
bradyrestdata=features((label(:,4)==0),:);
bradydataf=[bradydata; bradyrestdata];
bradylabels=[ones(1,size(bradydata,1)),zeros(1,size(bradyrestdata,1))]';
modelbrady = fitensemble(bradydataf,bradylabels,'RusBoost',30,t,'type','classification');
modelbrady=compact(modelbrady);

% PAC & SVPB MODEL
pacdata=features((label(:,13)==1)|(label(:,24)==1),:);
pacrestdata=features((label(:,13)==0)&(label(:,24)==0),:);
pacdataf=[pacdata; pacrestdata];
paclabels=[ones(1,size(pacdata,1)),zeros(1,size(pacrestdata,1))]';
modelpac = fitensemble(pacdataf,paclabels,'RusBoost',30,t,'type','classification');
modelpac=compact(modelpac);

% PVC & VPB MODEL
pvcdata=features((label(:,14)==1)|(label(:,27)==1),:);
pvcrestdata=features((label(:,14)==0)&(label(:,27)==0),:);
pvcdataf=[pvcdata; pvcrestdata];
pvclabels=[ones(1,size(pvcdata,1)),zeros(1,size(pvcrestdata,1))]';
modelpvc = fitensemble(pvcdataf,pvclabels,'RusBoost',30,t,'type','classification');
modelpvc=compact(modelpvc);

% SA MODEL
sadata=features((label(:,20)==1),:);
sarestdata=features((label(:,20)==0),:);
sadataf=[sadata; sarestdata];
salabels=[ones(1,size(sadata,1)),zeros(1,size(sarestdata,1))]';
modelsa = fitensemble(sadataf,salabels,'RusBoost',30,t,'type','classification');
modelsa=compact(modelsa);

% SB MODEL
sbdata=features((label(:,21)==1),:);
sbrestdata=features((label(:,21)==0),:);
sbdataf=[sbdata; sbrestdata];
sblabels=[ones(1,size(sbdata,1)),zeros(1,size(sbrestdata,1))]';
modelsb = fitensemble(sbdataf,sblabels,'RusBoost',30,t,'type','classification');
modelsb=compact(modelsb);

% STACH MODEL
stachdata=features((label(:,23)==1),:);
stachrestdata=features((label(:,23)==0),:);
stachdataf=[stachdata; stachrestdata];
stachlabels=[ones(1,size(stachdata,1)),zeros(1,size(stachrestdata,1))]';
modelstach = fitensemble(stachdataf,stachlabels,'RusBoost',30,t,'type','classification');
modelstach=compact(modelstach);

% IAVB MODEL
iavbdata=features((label(:,1)==1),:);
iavbrestdata=features((label(:,1)==0),:);
iavbdataf=[iavbdata; iavbrestdata];
iavblabels=[ones(1,size(iavbdata,1)),zeros(1,size(iavbrestdata,1))]';
modeliavb = fitensemble(iavbdataf,iavblabels,'RusBoost',30,t,'type','classification');
modeliavb=compact(modeliavb);

%CRBBB & RBBB MODEL
crbbbdata=features((label(:,5)==1)|(label(:,19)==1),:);
crbbbrestdata=features((label(:,5)==0)&(label(:,19)==0),:);
crbbbdataf=[crbbbdata; crbbbrestdata];
crbbblabels=[ones(1,size(crbbbdata,1)),zeros(1,size(crbbbrestdata,1))]';
modelcrbbb = fitensemble(crbbbdataf,crbbblabels,'RusBoost',30,t,'type','classification');
modelcrbbb=compact(modelcrbbb); 

%IRBBB MODEL
irbbbdata=features((label(:,6)==1),:);
irbbbrestdata=features((label(:,6)==0),:);
irbbbdataf=[irbbbdata; irbbbrestdata];
irbbblabels=[ones(1,size(irbbbdata,1)),zeros(1,size(irbbbrestdata,1))]';
modelirbbb = fitensemble(irbbbdataf,irbbblabels,'RusBoost',30,t,'type','classification');
modelirbbb=compact(modelirbbb);

%LANFB MODEL
lanfbdata=features((label(:,7)==1),:);
lanfbrestdata=features((label(:,7)==0),:);
lanfbdataf=[lanfbdata; lanfbrestdata];
lanfblabels=[ones(1,size(lanfbdata,1)),zeros(1,size(lanfbrestdata,1))]';
modellanfb = fitensemble(lanfbdataf,lanfblabels,'RusBoost',30,t,'type','classification');
modellanfb=compact(modellanfb);

%LAD MODEL
laddata=features((label(:,8)==1),:);
ladrestdata=features((label(:,8)==0),:);
laddataf=[laddata; ladrestdata];
ladlabels=[ones(1,size(laddata,1)),zeros(1,size(ladrestdata,1))]';
modellad = fitensemble(laddataf,ladlabels,'RusBoost',30,t,'type','classification');
modellad=compact(modellad);

%LBBB MODEL
lbbbdata=features((label(:,9)==1),:);
lbbbrestdata=features((label(:,9)==0),:);
lbbbdataf=[lbbbdata; lbbbrestdata];
lbbblabels=[ones(1,size(lbbbdata,1)),zeros(1,size(lbbbrestdata,1))]';
modellbbb = fitensemble(lbbbdataf,lbbblabels,'RusBoost',30,t,'type','classification');
modellbbb=compact(modellbbb);

%LQRSV MODEL
lqrsvdata=features((label(:,10)==1),:);
lqrsvrestdata=features((label(:,10)==0),:);
lqrsvdataf=[lqrsvdata; lqrsvrestdata];
lqrsvlabels=[ones(1,size(lqrsvdata,1)),zeros(1,size(lqrsvrestdata,1))]';
modellqrsv = fitensemble(lqrsvdataf,lqrsvlabels,'RusBoost',30,t,'type','classification');
modellqrsv=compact(modellqrsv);

%NSIVCB MODEL
nsivcbdata=features((label(:,11)==1),:);
nsivcbrestdata=features((label(:,11)==0),:);
nsivcbdataf=[nsivcbdata; nsivcbrestdata];
nsivcblabels=[ones(1,size(nsivcbdata,1)),zeros(1,size(nsivcbrestdata,1))]';
modelnsivcb = fitensemble(nsivcbdataf,nsivcblabels,'RusBoost',30,t,'type','classification');
modelnsivcb=compact(modelnsivcb);

%PR MODEL
prdata=features((label(:,12)==1),:);
prrestdata=features((label(:,12)==0),:);
prdataf=[prdata; prrestdata];
prlabels=[ones(1,size(prdata,1)),zeros(1,size(prrestdata,1))]';
modelpr = fitensemble(prdataf,prlabels,'RusBoost',30,t,'type','classification');
modelpr=compact(modelpr);

% LPR MODEL
lprdata=features((label(:,15)==1),:);
lprrestdata=features((label(:,15)==0),:);
lprdataf=[lprdata; lprrestdata];
lprlabels=[ones(1,size(lprdata,1)),zeros(1,size(lprrestdata,1))]';
modellpr = fitensemble(lprdataf,lprlabels,'RusBoost',30,t,'type','classification');
modellpr=compact(modellpr);

% LQT MODEL
lqtdata=features((label(:,16)==1),:);
lqtrestdata=features((label(:,16)==0),:);
lqtdataf=[lqtdata; lqtrestdata];
lqtlabels=[ones(1,size(lqtdata,1)),zeros(1,size( lqtrestdata,1))]';
modellqt = fitensemble(lqtdataf,lqtlabels,'RusBoost',30,t,'type','classification');
modellqt=compact(modellqt);

% QAB MODEL
qabdata=features((label(:,17)==1),:);
qabrestdata=features((label(:,17)==0),:);
qabdataf=[qabdata; qabrestdata];
qablabels=[ones(1,size(qabdata,1)),zeros(1,size( qabrestdata,1))]';
modelqab = fitensemble(qabdataf,qablabels,'RusBoost',30,t,'type','classification');
modelqab=compact(modelqab);

% RAD MODEL
raddata=features((label(:,18)==1),:);
radrestdata=features((label(:,18)==0),:);
raddataf=[raddata; radrestdata];
radlabels=[ones(1,size(raddata,1)),zeros(1,size( radrestdata,1))]';
modelrad = fitensemble(raddataf,radlabels,'RusBoost',30,t,'type','classification');
modelrad=compact(modelrad);

% SNR MODEL
snrdata=features((label(:,22)==1),:);
snrrestdata=features((label(:,22)==0),:);
snrdataf=[snrdata; snrrestdata];
snrlabels=[ones(1,size(snrdata,1)),zeros(1,size(snrrestdata,1))]';
modelsnr = fitensemble(snrdataf,snrlabels,'RusBoost',30,t,'type','classification');
modelsnr=compact(modelsnr);

% TAB MODEL
tabdata=features((label(:,25)==1),:);
tabrestdata=features((label(:,25)==0),:);
tabdataf=[tabdata; tabrestdata];
tablabels=[ones(1,size(tabdata,1)),zeros(1,size(tabrestdata,1))]';
modeltab = fitensemble(tabdataf,tablabels,'RusBoost',30,t,'type','classification');
modeltab=compact(modeltab);

% TINV MODEL
tinvdata=features((label(:,26)==1),:);
tinvrestdata=features((label(:,26)==0),:);
tinvdataf=[tinvdata; tinvrestdata];
tinvlabels=[ones(1,size(tinvdata,1)),zeros(1,size(tinvrestdata,1))]';
modeltinv = fitensemble(tinvdataf,tinvlabels,'RusBoost',30,t,'type','classification');
modeltinv=compact(modeltinv);

%others
othdata=features(find(sum(label,2)==0),:);
othrestdata=features(find(sum(label,2)>0),:);
othdataf=[othdata; othrestdata];
othlabels=[ones(1,size(othdata,1)),zeros(1,size(othrestdata,1))]';
modeloth = fitensemble(othdataf,othlabels,'RusBoost',30,t,'type','classification');
modeloth=compact(modeloth);

model{1}=modeliavb;
model{2}=modelaf;
model{3}=modelafl;
model{4}=modelbrady;
model{5}=modelcrbbb;
model{6}=modelirbbb;
model{7}=modellanfb;
model{8}=modellad;
model{9}=modellbbb;
model{10}=modellqrsv;
model{11}=modelnsivcb;
model{12}=modelpr;
model{13}=modelpac;
model{14}=modelpvc;
model{15}=modellpr;
model{16}=modellqt;
model{17}=modelqab;
model{18}=modelrad;
model{19}=modelsa;
model{20}=modelsb;
model{21}=modelsnr;
model{22}=modelstach;
model{23}=modeltab;
model{24}=modeltinv;
model{25}=modeloth;


save_12_ECG_model(model,output_directory,classes);

end

function save_12_ECG_model(model,output_directory,classes)
% Save results.
tmp_file = 'finalized_model.mat';
filename=fullfile(output_directory,tmp_file);
save(filename,'model','classes','-v7.3');


disp('Done.')
end


% find unique number of classes
function classes = get_classes(input_directory,files)

classes={};
num_files = length(files);
k=1;
for i = 1:num_files
    g = strrep(files{i},'.mat','.hea');
    input_file = fullfile(input_directory, g);
    fid=fopen(input_file);
    tline = fgetl(fid);
    tlines = cell(0,1);
    
    while ischar(tline)
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
        if startsWith(tline,'#Dx')
            tmp = strsplit(tline,': ');
            tmp_c = strsplit(tmp{2},',');
            for j=1:length(tmp_c)
                idx2 = find(strcmp(classes,tmp_c{j}));
                if isempty(idx2)
                    classes{k}=tmp_c{j};
                    k=k+1;
                end
            end
            break
        end
    end
    
    fclose(fid);
    
end
classes=sort(classes);
end

function [data,tlines] = load_challenge_data(filename)

% Opening header file
fid=fopen([filename '.hea']);

if (fid<=0)
    disp(['error in opening file ' filename]);
end

tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

f=load([filename '.mat']);

try
    data = f.val;
catch ex
    rethrow(ex);
end

end
