function features = get_12ECG_features(data, header_data)

       % addfunction path needed
        addpath(genpath('Tools/'))
       % load('HRVparams_12ECG','HRVparams')
        temp1=load('feature2l1');
        temp2=load('feature2l2');
        temp3=load('feature2l3');

	% read number of leads, sample frequency and gain from the header.	

	[recording,Total_time,num_leads,Fs,gain,age,sex]=extract_data_from_header(header_data);
    [QRSlead1,~,~]=qrs_detect2(data(1,:)',0.25,0.6,Fs,[],[],0);
    [QRSavr,~,~]=qrs_detect2(data(4,:)',0.25,0.6,Fs,[],[],0);
    filtdata=BP_filter_ECG(data(1,:),Fs);
     fblead1=feat_29_2020_a(data(1,:),Fs);
     feat_st_p=st_p_stats(filtdata,QRSlead1,Fs);
     pr_feat=pr_stats(filtdata,QRSlead1,Fs);
     st_avr=st_p_stats_avr(data(4,:),QRSavr,Fs);
     tqwtlead1=tqwt_analysis(data(1,:),QRSlead1,Fs);
     tqwtavr=tqwt_analysis(data(4,:),QRSavr,Fs);
     tqwtv1=tqwt_analysis(data(7,:),QRSlead1,Fs);
     stv1=st_p_stats_avr(data(7,:),QRSlead1,Fs);
     stv2=st_p_stats_avr(data(8,:),QRSlead1,Fs);
     stv4=st_p_stats_avr(data(10,:),QRSlead1,Fs);
     stavl=st_p_stats_avr(data(5,:),QRSlead1,Fs);
     tqwtv2=tqwt_analysis(data(8,:),QRSlead1,Fs);
     tqwtv4=tqwt_analysis(data(10,:),QRSlead1,Fs);
     tqwtavl=tqwt_analysis(data(5,:),QRSlead1,Fs);
     distl1=get_distfeat(data(1,:),Fs,temp1.feature2);
     distl2=get_distfeat(data(2,:),Fs,temp2.feature2);
     distl3=get_distfeat(data(3,:),Fs,temp3.feature2);
     features=[fblead1 feat_st_p pr_feat st_avr tqwtlead1 tqwtavr tqwtv1 stv1 stv2 stv4 stavl tqwtv2 tqwtv4 tqwtavl distl1 distl2 distl3];

end

function [filt_signal1] = BP_filter_ECG(ecg,fs)
d = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.05,'HalfPowerFrequency2',100, ...
    'SampleRate',fs);            
    filt_signal1=filtfilt(d,ecg);   
end

