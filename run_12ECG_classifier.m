function [score, label,classes] = run_12ECG_classifier(data,header_data, loaded_model)


	model=loaded_model.model;
	classes=loaded_model.classes;

    num_classes = length(classes);

    label = zeros([1,num_classes]);
    score = ones([1,num_classes]);
    
    % Use your classifier here to obtain a label and score for each class.
    features = get_12ECG_features(data,header_data);
[~,prediavbsc] = predict(model{1},features);
[~,predafsc] = predict(model{2},features);
[~,predaflsc] = predict(model{3},features);
[~,predbradysc] = predict(model{4},features);
[~,predcrbbbsc] = predict(model{5},features);
[~,predirbbbsc] = predict(model{6},features);
[~,predlanfbsc] = predict(model{7},features);
[~,predladsc] = predict(model{8},features);
[~,predlbbbsc] = predict(model{9},features);
[~,predlqrsvsc] = predict(model{10},features);
[~,prednsivcbsc] = predict(model{11},features);
[~,predprsc] = predict(model{12},features);
[~,predpacsc] = predict(model{13},features);
[~,predpvcsc] = predict(model{14},features);
[~,predlprsc] = predict(model{15},features);
[~,predlqtsc] = predict(model{16},features);
[~,predqabsc] = predict(model{17},features);
[~,predradsc] = predict(model{18},features);
%[~,predrbbbsc] = predict(model{5},features);
[~,predsasc] = predict(model{19},features);
[~,predsbsc] = predict(model{20},features);
[~,predsnrsc] = predict(model{21},features);
[~,predstachsc] = predict(model{22},features);
%[~,predsvpbsc] = predict(model{13},features);
[~,predtabsc] = predict(model{23},features);
[~,predtinvsc] = predict(model{24},features);
%[~,predvpbsc] = predict(model{14},features);
[~,predothsc] = predict(model{25},features);


    score1 = [prediavbsc(1,2)/sum(prediavbsc) predafsc(1,2)/sum(predafsc) predaflsc(1,2)/sum(predaflsc) predbradysc(1,2)/sum(predbradysc) predcrbbbsc(1,2)/sum(predcrbbbsc) predirbbbsc(1,2)/sum(predirbbbsc) predlanfbsc(1,2)/sum(predlanfbsc,2) predladsc(:,2)./sum(predladsc) predlbbbsc(1,2)/sum(predlbbbsc) predlqrsvsc(1,2)/sum(predlqrsvsc) prednsivcbsc(1,2)/sum(prednsivcbsc,2) predprsc(1,2)/sum(predprsc) predpacsc(1,2)/sum(predpacsc) predpvcsc(1,2)/sum(predpvcsc,2) predlprsc(1,2)/sum(predlprsc) predlqtsc(1,2)/sum(predlqtsc,2) predqabsc(1,2)/sum(predqabsc) predradsc(1,2)/sum(predradsc) predsasc(1,2)/sum(predsasc) predsbsc(1,2)/sum(predsbsc) predsnrsc(1,2)/sum(predsnrsc) predstachsc(1,2)/sum(predstachsc) predtabsc(1,2)/sum(predtabsc) predtinvsc(1,2)/sum(predtinvsc) predothsc(1,2)/sum(predothsc)];		
    [~,idx] = max (score1);
if idx==5
idx=[5 19];
elseif idx==13
idx=[13 24];
elseif idx==14
idx=[14 27];
elseif idx==19
idx=20;
elseif idx==20
idx=21;
elseif idx==21
idx=22;
elseif idx==22
idx=23;
elseif idx==23
idx=25;
elseif idx==24
idx=26;
end
if idx==25
else
    label(idx)=1;
end
score=[prediavbsc(1,2)/sum(prediavbsc) predafsc(1,2)/sum(predafsc) predaflsc(1,2)/sum(predaflsc) predbradysc(1,2)/sum(predbradysc) predcrbbbsc(1,2)/sum(predcrbbbsc) predirbbbsc(1,2)/sum(predirbbbsc) predlanfbsc(1,2)/sum(predlanfbsc,2) predladsc(:,2)./sum(predladsc) predlbbbsc(1,2)/sum(predlbbbsc) predlqrsvsc(1,2)/sum(predlqrsvsc) prednsivcbsc(1,2)/sum(prednsivcbsc,2) predprsc(1,2)/sum(predprsc) predpacsc(1,2)/sum(predpacsc) predpvcsc(1,2)/sum(predpvcsc,2) predlprsc(1,2)/sum(predlprsc) predlqtsc(1,2)/sum(predlqtsc,2) predqabsc(1,2)/sum(predqabsc) predradsc(1,2)/sum(predradsc) predcrbbbsc(1,2)/sum(predcrbbbsc) predsasc(1,2)/sum(predsasc) predsbsc(1,2)/sum(predsbsc) predsnrsc(1,2)/sum(predsnrsc) predstachsc(1,2)/sum(predstachsc) predpacsc(1,2)/sum(predpacsc) predtabsc(1,2)/sum(predtabsc) predtinvsc(1,2)/sum(predtinvsc) predpvcsc(1,2)/sum(predpvcsc,2)];		
  
end



