function iris

    input = load("iris_in.csv");    % dim: 150*4
    target = load("iris_out.csv");  % dim: 150*1
    
    % initialize the weight matrix
    output_matrix = rand(12,1); % (12*1)
    hidden_matrix = rand(4,12); % (4*12)
    
    RMSE = zeros(100,1);
    
    lr = 0.025;
    % Training
    for epoch=1:100
        t = zeros(75, 1);
        for idx=1:75
            x = input(idx, :);  % (1*4)
            y_ = target(idx);   % (1*1)
            sigma_hid = x * hidden_matrix;  % (1*12)
            a_hid = logsig(sigma_hid);      % (1*12)
            sigma_out = a_hid * output_matrix;  % (1*1)
            y = purelin(sigma_out);         % (1*1) % predict result; a_out 
            % calculate error and delta
            error = y_ - y;     % (1*1)
            delta_out = error;  % (1*1)
            delta_hid = delta_out * dpurelin(sigma_out) * output_matrix;% (1*1)(1*1)(12*1)
            % update weights
            output_matrix = output_matrix + lr * (a_hid' * (delta_out' .* dpurelin(sigma_out)));
            hidden_matrix = hidden_matrix + lr * (x' * (delta_hid' .* dlogsig(a_hid, logsig(sigma_hid))));
            t(idx, :) = error.^2;
        end
        RMSE(epoch) = sqrt(sum(t)/75);
        fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, RMSE(epoch));
    end
    
    % display training process
    fprintf('\nTotal number of epochs: %g\n', epoch);
    fprintf('Final RMSE: %g\n', RMSE(epoch));
    plot(1:epoch,RMSE(1:epoch));
    legend('Training');
    ylabel('RMSE');xlabel('Epoch');
    
    
    % Testing
    num_correct=0;
    for idx=76:length(input)
        x = input(idx, :);  % (1*4)
        y_ = target(idx);   % (1*1)
        y = purelin(logsig(x * hidden_matrix) * output_matrix);
        if y > y_ - 0.5 && y <= y_ + 0.5
            num_correct = num_correct + 1;
        end
    end    
    Tot_Percent= (num_correct) / (length(input)-75);
    fprintf('Test correct percent: %f\n', Tot_Percent);
end
