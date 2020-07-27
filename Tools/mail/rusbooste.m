function [ PIndices ] = rusbooste( adata )
%UNTITLED20 Summary of this function goes here
%   Detailed explanation goes here
% best setting Bag' 
477 and minleaf = 2............
covtype=adata(:,1:47);

Y=adata(:,48);%,,2,13,7,14,75
%group=full(ind2vec(groupo'));
part = crossvalind('Kfold',Y,5);
%models={};\

for i = 1:5%469
    test = (part == i); trainl = ~test;
    
    t = ClassificationTree.template('minleaf',2);%'MinParent',5,'NVarToSample',15
    % ClassNames=[1 2 3 4];
    % cost.ClassNames = ClassNames;
    % cost.ClassificationCosts = [0 10 250 70;10 0 1 1 ;250 1 0 1;70 1 1 0];
    rusTree = fitensemble(covtype(trainl,:),Y(trainl),'Bag',477,t,'type','classification',...
         'nprint',500);
    Yfit = predict(rusTree,covtype(test,:));
    %models{i,1}=rusTree;
    %[c,cm,ind,per] = confusion(Y(test)',Yfit');%for binary
    plotconfusion(full(ind2vec(Y(test)',4)),full(ind2vec(Yfit',4)));
    [c,cm1,ind,per] = confusion(full(ind2vec(Y(test)',4)),full(ind2vec(Yfit',4)));
   cm=cm1';
    %[c,cm,ind,per] = confusion(full(ind2vec(Y(test)',3)),full(ind2vec(Yfit',3)));
    d(i)=(1-c);
    
    %plotconfusion(ind2vec(Yfit'),ind2vec(Y(test)'))%correct expression
    %plotconfusion(Yfit',Y(test)')%binary
    Faf(i)=2*cm(1,1)/(sum(cm(:,1))+sum(cm(1,:)));
    Fn(i)=2*cm(2,2)/(sum(cm(:,2))+sum(cm(2,:)));
    Fo(i)=2*cm(3,3)/(sum(cm(:,3))+sum(cm(3,:)));
    Fnoise(i)=2*cm(4,4)/(sum(cm(:,4))+sum(cm(4,:)));
    
    
end
avrd=nanmean(d);

%fprintf('Percentage Correct Classification   : %f%%\n', avrd*100);
%fprintf('Percentage Correct Classification   : %f%%\n', 100*mean([mean(Far) mean(Fnar) ]));
PIndices= [mean(Faf) mean(Fn)  mean(Fo) mean(Fnoise) avrd].*100;
end


