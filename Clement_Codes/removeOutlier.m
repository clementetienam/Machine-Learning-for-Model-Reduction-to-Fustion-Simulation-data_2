function mat = removeOutlier(train2)
train2=train;
    [mv mi] = max(train2(:,1:end));
    x = mode(mv);
    train2(x, :)=[];
    [mv mi] = min(train2(:,1:end));
    x = mode(mv);
    train2(x, :)=[];
    mat = train2;