function [feat] = get_distfeat(data,fs,feature2)
%load feature2l3;
if fs==500
else
    ecg = resample(data,500,fs);
end
fs=500;
ar_order=8;
samp_len=250;
ecg=data;
[QRS,sign,en_thres] = qrs_detect2(ecg',0.25,0.6,fs);%Detecting QRS ( Note: Included as it is from the sample file)
for i=1:1:size(QRS,2)-3
ecg_seg2(i,:)=ecg((QRS(1,i+1)-(0.2*fs)):(QRS(1,i+1)-(0.2*fs)+samp_len-1));
end
try
for i=1:1:size(ecg_seg2,1)
feature1(i,:) = getarfeat(ecg_seg2(i,:)',ar_order,samp_len,samp_len);
end
k=1;
for i=1:1:20
for j=1:1:size(feature1,1)
pf2=abs(fft((feature1(j,:))).^2);
pf1=abs(fft((feature2(i,:))).^2);
d_itar(k,:) =distitar(feature2(i,:),feature1(j,:),'d');
d_itpf(k,:)=distitpf(pf1,pf2,'d');
% d_eu(k,:)=disteusq(x,y,mode,w);
d_itsar(k,:)=distisar(feature2(i,:),feature1(j,:),'d');
d_copf(k,:)=distchpf(pf1,pf2,'d');
d_coar(k,:)=distchar(feature2(i,:),feature1(j,:),'d');
d_itspf(k,:)=distispf(pf1,pf2,'d');
k=k+1;
end
end
feat=[mean(d_coar) mean(d_itar) mean(d_itsar) mean(d_copf) mean(d_itpf) mean(d_itspf)];
catch
    feat=[1 1 1 1 1 1];
end
end
