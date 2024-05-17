function iris_oneHotEncoding
    % one-hot-encoding version

    input = load("iris_in.csv");            % dim: 150*4
    target_lbl = load("iris_out.csv");      % dim: 150*1
    target = zeros(size(target_lbl,1), 3);    % dim: 150*3
    
    % transfer target_lbl to one-hot-encoding
    for t=1:size(target_lbl,1)
        switch target_lbl(t)
            case 1
                target(t, 1) = 1;
            case 2
                target(t, 2) = 1;
            case 3
                target(t, 3) = 1;
        end
    end


    % initialize the weight matrix
    output_matrix = rand(12,1); % (12*3)
    hidden_matrix = rand(4,12); % (4*12)
    
    RMSE = zeros(100,1);
    
    lr = 0.025;
    % Training
    for epoch=1:1
        t = zeros(75, 1);
        for idx=1:75
            x = input(idx, :);  % (1*4)
            y_ = target(idx);   % (1*3)
            sigma_hid = x * hidden_matrix;  % (1*12)
            a_hid = logsig(sigma_hid);      % (1*12)
            sigma_out = a_hid * output_matrix;  % (1*3)=(1*12)(12*3)
            y = purelin(sigma_out);         % (1*3) % predict result; a_out 
            % calculate error and delta
            error = y_ - y;     % (1*3)
            delta_out = sqrt(error.^2);  % (1*3)
            disp(delta_out)
            delta_hid = delta_out * output_matrix' .* dlogsig(sigma_hid, a_hid); %(1*12) = (1*3)(3*12).*(1*12)
            % update weights
            output_matrix = output_matrix - lr * (a_hid' * (delta_out .* dpurelin(sigma_out))); %(12*1)*(1*3)
            hidden_matrix = hidden_matrix - lr * (x' * (delta_hid .* dlogsig(sigma_hid, a_hid)));
            t(idx, :) = sum(error.^2);
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
    y_lbls = [];
    for idx=76:length(input)
        x = input(idx, :);  % (1*4)
        t_lbl = target_lbl(idx);   % (1*1)
        y = purelin(logsig(x * hidden_matrix) * output_matrix); % (1*3)
        [~, y_lbl] = max(y);
        y_lbls(idx-75, :) = y_lbl;
        if y_lbl == t_lbl
            num_correct = num_correct + 1;
        end
    end    
    Tot_Percent= (num_correct) / (length(input)-75);
    fprintf('Test correct percent: %f\n', Tot_Percent);
    % y_lbls;
end
