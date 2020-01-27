function [eer, eer_i] = roc2eer(rocdata)

    fpr = rocdata(:,1);
    tpr = rocdata(:,2);

    fnr = 1 - tpr;

    [~, eer_i] = min(abs(fnr - fpr));
    eer = fpr(eer_i);

end