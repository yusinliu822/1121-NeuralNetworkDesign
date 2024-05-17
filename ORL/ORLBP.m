function [Recognition,Final,input]=ORLBP

%% PCA LDA
people=40;
principlenum=30;
withinsample=5;
HiddenNeuroNum=60;

ouputLR=0.1;
hiddenLR=0.1;

FFACE=[];

for k=1:1:people
    for m=1:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
        matchX=imread(matchstring);
        matchX=double(matchX);
        
        if(k==1 && m==1)
            [row,col]=size(matchX);
        end
        
        matchtempF=[];
        
        for n=1:row
            matchtempF=[matchtempF, matchX(n,:)];
        end
        
        FFACE=[FFACE; matchtempF];
    end
end

% FFACE = FFACE - TotalMeanFace;
TotalMeanFace = mean(FFACE);
zeromeanTotalFace=FFACE;
for i=1:1:withinsample*people
    for j=1:1:(row)*(col)
        zeromeanTotalFace(i,j)=zeromeanTotalFace(i,j)-TotalMeanFace(j);
    end
end

SST = zeromeanTotalFace'*zeromeanTotalFace; %covarianve matrix
%SST=cov(zeromeanTotalFace)

[eigVector,latent] = eig(SST);

eigvalue=diag(latent);

[junk, index] = sort(eigvalue, 'descend');

eigVector =eigVector(:, index);
eigvalue = eigvalue(index);

projectPCA=eigVector(:,1:principlenum);

pcaTotalFACE = zeromeanTotalFace * projectPCA;
% end of PCA

for i=1:withinsample:withinsample*people
    withinFACE = pcaTotalFACE(i:i+withinsample-1,:);
    if(i==1)
        SW=withinFACE'*withinFACE;
        ClassMean=mean(withinFACE);
    end
    if(i>1)
        SW=SW+withinFACE'*withinFACE;
        ClassMean=[ClassMean;mean(withinFACE)];
    end
end
pcaTotalmean= mean(pcaTotalFACE);

SB = ClassMean'*ClassMean;

[eigvector, eigvalue] = eig(inv(SW)*SB);
eigvalue = diag(eigvalue);
[junk,index]=sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvetor = eigvector(:,index);
prototypeFACE=pcaTotalFACE*eigvector(:,1:30);
prototypeFACE=prototypeFACE/max(abs(prototypeFACE(:)));

input = prototypeFACE;


%% Initialization

outputmatrix = zeros(HiddenNeuroNum,people);
for i=1:HiddenNeuroNum
     for j=1:people
        outputmatrix(i,j)=rand;
     end
end

hiddenmatrix = zeros(principlenum,HiddenNeuroNum);
for i=1:principlenum
    for j=1:HiddenNeuroNum
        hiddenmatrix(i,j)=rand;
    end
end

outputbias = zeros(1,people);
for i=1:people
    outputbias(1,i)=rand;
end

hiddenbias = zeros(1,HiddenNeuroNum);
for i=1:HiddenNeuroNum
    hiddenbias(1,i)=rand;
end

target=[];
for i=1:1:people
     target=[target,i,i,i,i,i]; 
end
target = double(target'==1:people);

%% shuffling
sh = randperm(200);
input = input(sh,:);
trainingtarget=target(sh,:);

for epoch=1:200
    t=[];
    for iter=1:200
        %forward
        hiddensigma = input(iter,:)*hiddenmatrix+hiddenbias;
        hiddennet = logsig(hiddensigma);
        
        outputsigma = hiddennet*outputmatrix+outputbias;
        outputnet = sigmoid(outputsigma);
        
        %backward
        %output layer delta
        doutputnet=dsigmoid(outputnet);
        deltaoutput = ((trainingtarget(iter,:)-outputnet).*doutputnet)';
        %hidden layer delta
        transfer = dlogsig(hiddensigma,hiddennet);
        tempdelta = outputmatrix*deltaoutput;
        deltahidden = tempdelta.*(transfer');
        %output weight update
        outputmatrix = outputmatrix + 0.3*(deltaoutput*hiddennet)';
        %hidden weight update
        hiddenmatrix = hiddenmatrix + 0.3*(deltahidden*input(iter,:))';
        %bias update
        outputbias = outputbias + 0.3*deltaoutput';
        hiddenbias = hiddenbias + 0.3*deltahidden';
        %error
        error = trainingtarget(iter,:)-outputnet;
        error = error.^2;
        error = sqrt(sum(error));
        t=[t;error];
        
    end
    RMSE(epoch) = sqrt(sum(t)/200);
    fprintf('epoch %d  RMSE = %.3f\n' ,epoch, sqrt(sum(t)/200));
end
%%
perdiction=[];
correct=0;
total=0;
testFACE=[];   
for k=1:1:people
    for m=2:2:10
        matchstring=['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
        matchX=imread(matchstring);
        matchX=double(matchX);
        
        if(k==1 && m==2)
            [row,col]=size(matchX);
        end
        
        matchtempF=[];
        
        for n=1:row
            matchtempF=[matchtempF, matchX(n,:)];
        end
        
        testFACE=[testFACE; matchtempF];
    end
end
% testFACE = testFACE - TotalMeanFace;
testFACEmean = mean(testFACE);
testFACE = testFACE-testFACEmean;
pcatestFACE=testFACE*projectPCA;
projecttestFACE=pcatestFACE*eigvector(:,1:30);

test = projecttestFACE;
test=test/max(abs(test(:)));


correct=0;
Final=[];
for i=1:1:200
    hiddensigma = test(i,:)*hiddenmatrix+hiddenbias;
    hiddennet = logsig(hiddensigma);

    outputsigma = hiddennet*outputmatrix+outputbias;
    outputnet = sigmoid(outputsigma);
    Final=[Final;outputnet];
    [trash,pred]=max(outputnet);
    [trash,true]=max(target(i,:));
    if pred==true
        correct=correct+1;
    end
end
Recognition=correct/200;
end

