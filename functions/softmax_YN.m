function [out_Vector]=softmax_YN(in_Vector) 
     exp_in_Vector=exp(in_Vector);
     out_Vector=exp_in_Vector/sum(exp_in_Vector);
end