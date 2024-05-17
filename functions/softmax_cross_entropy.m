function [ce] = softmax_cross_entropy(labels , logits)
    q = softmax(logits);
    q = -log(q);
    p = labels;
    ce = p .* q;
end

function [out_Vector]=softmax(in_Vector) 
     exp_in_Vector=exp(in_Vector);
     out_Vector=exp_in_Vector/sum(exp_in_Vector);
end