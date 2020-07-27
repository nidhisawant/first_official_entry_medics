function [Kurt]= tqwt_analysis(ndata,QRS,fs)
try
x=[1 3 7];
sb=tqwt_radix2(ndata,x(1),x(2),x(3));

%for kk=1:x(3)+1
      %Ffactor(kk)= kurtosis(sb{1,kk})./max(abs(sb{1,kk}));%(var(abs(sb{1,kk}))./mean(abs(sb{1,kk}))./(max(abs(sb{1,kk}))));
%end
%mask=Ffactor<threshold;%skewness(sb{1,kk});%
 mask=logical([0 0 0 0 0 0 0 1]);
 for kk=1:x(3)+1
      selected_sb{1,kk}= sb{1,kk}*double(mask(kk));
 end
  
 rec_sig=itqwt_radix2(selected_sb,x(1),x(2),length(ndata));
 %fratio=(var(abs(ndata))./mean(abs(ndata))./(max(abs(ndata))))./(var(abs(rec_sig))./mean(abs(rec_sig))./(max(abs(rec_sig))));
%eratio=sum((ndata./(max(ndata)-min(ndata))).^2)-sum((rec_sig./(max(rec_sig)-min(rec_sig))).^2);
 %Kurt=kurtosis(rec_sig);
plot(ndata./(max(ndata)-min(ndata)))
hold on
plot(rec_sig./(max(rec_sig)-min(rec_sig)))

corcf=corrcoef(rec_sig,ndata);


xxx=ndata(QRS)-rec_sig(QRS);
Kurt=[ststats2(xxx,1) ststats2(ndata,fs)./ststats2(rec_sig,fs)   corcf(1,2) ] ;%1 4 7 11 12
 %qrj 1 3 7
  %Ffact= (var(abs(rec_sig))./mean(abs(rec_sig))./(max(abs(rec_sig))));

catch
    Kurt=NaN.*ones(1,33);
end

end
function[feat]=ststats2(x,fs)

   
        
    st_median = median(x);
    st_kurto = kurtosis(x);
    st_skew = skewness(x);
    st_mean = mean(x);
    st_mode = mode(x);
    st_max=max(x);
    st_min=min(x);
    st_var=var(x);
    energy=sum(x.^2);
    e1=wentropy(x,'norm',4);
    e2=wentropy(x,'sure',0.06);
    e3=wentropy(x,'norm',3);
    
    ffac=(std(diff(diff(x))).*std(x))./std(diff(x)).^2;
    
    %bandpower(x,1000,[50 150])
    f1=obw(x,fs);
    f2=meanfreq(x,fs);
    f3=medfreq(x,fs);

    feat=[ f1 f2 f3 ffac e1 e2 e3 energy st_var st_median, st_kurto, st_skew, st_mean, st_mode,st_max, st_min ];
    
    %[1 2 4 5 6 7 9] useful for normal
 
end