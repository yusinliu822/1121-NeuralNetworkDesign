function [result]=dsigmoid(X)

% sigmoid'(X) = sigmoid(X) * (1-sigmoid(X))
% Reference: http://www.ai.mit.edu/courses/6.892/lecture8-html/sld015.htm
result = sigmoid(X).*(1-sigmoid(X));

end
