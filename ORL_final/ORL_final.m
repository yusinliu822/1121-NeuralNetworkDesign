function ORL_final
    data = load("ORL_PCA_MaxMin_Data.mat");
    train_data = data.TrainORL; % (200*65)
    test_data = data.TestORL;   % (200*65)
    target_lbl = data.Target;       % (200*1)
    % hidden_layer = data.HidLayer;   %(65*110)
    % output_layer = data.OutLayer;   %(110*40)
    % output = data.OUTput        % (40*200)
    
    SAVE = true;
    VALID = true;
    NUM_TRAIN_DATA = 200;
    NUM_VALID_DATA = 20;
    NUM_TEST_DATA = 200;
    DIM_IN = 65;
    DIM_HID = 110;  % number of hidden layer neuron
    DIM_OUT = 40;   % number of class / number of output layer neuron

    LR = 0.25;   % learning rate
    NUM_EPOCH = 500;

    % transfer target_lbl to one-hot-encoding
    target_onehot = zeros(NUM_TRAIN_DATA, DIM_OUT);   % (200*40)
    for t=1:size(target_lbl,1)
        target_onehot(t, target_lbl(t)) = 1;
    end

    % initialize the weight matrix
    hidden_matrix = randn(DIM_IN, DIM_HID);
    hidden_bias = randn(1, DIM_HID);
    output_matrix = randn(DIM_HID,DIM_OUT);
    output_bias = randn(1, DIM_OUT);

    % hidden_matrix = hidden_layer;
    % output_matrix = output_layer;

    % shuffling
    shuffle_idx = randperm(NUM_TRAIN_DATA);
    if VALID
        valid_shuffle_idx = shuffle_idx(1:NUM_VALID_DATA);
        shuffle_idx = shuffle_idx(NUM_VALID_DATA+1 : NUM_TRAIN_DATA);
        valid_input = train_data(valid_shuffle_idx, :);
        valid_target_lbl = target_lbl(valid_shuffle_idx, :);
        NUM_TRAIN_DATA = NUM_TRAIN_DATA - NUM_VALID_DATA;
    end
    input = train_data(shuffle_idx, :);
    target = target_onehot(shuffle_idx, :);


    % Training
    loss_per_epoch = zeros(NUM_EPOCH, 1);
    loss_per_step = zeros(NUM_EPOCH * NUM_TRAIN_DATA, 1);
    acc_per_epoch = zeros(NUM_EPOCH, 1);
    for epoch = 1:NUM_EPOCH
        losses = zeros(NUM_TRAIN_DATA, 1);
        for idx = 1:NUM_TRAIN_DATA
            x = input(idx, :);
            y_ = target(idx, :);
            
            % forward pass
            hidden_sigma = x * hidden_matrix + hidden_bias;
            hidden_net = logsig(hidden_sigma);
            
            output_sigma = hidden_net * output_matrix + output_bias;
            y = softmax(output_sigma);  % output_net

            % calculate loss with cross entropy
            loss = cross_entropy(y_, y);
            losses(idx, :) = loss;
            loss_per_step(idx+(epoch-1)*NUM_TRAIN_DATA, :) = loss;

            % backward pass
            delta_output = (y - y_)';
            d_hidden_net = dlogsig(hidden_sigma, hidden_net)';
            delta_hidden = (output_matrix * delta_output) .* d_hidden_net;

            % update weights & bias
            output_matrix = output_matrix - LR * (delta_output * hidden_net)';
            output_bias = output_bias - LR * delta_output';
            hidden_matrix = hidden_matrix - LR * (delta_hidden * x)';
            hidden_bias = hidden_bias - LR * delta_hidden';
        end
        loss_per_epoch(epoch) = sum(losses)/NUM_TRAIN_DATA;
        fprintf('epoch %.0f:  Loss = %.6f\n', epoch, loss_per_epoch(epoch));

        % validating
        if VALID
            num_correct=0;
            y_lbls = zeros(NUM_VALID_DATA, 1);
            for idx = 1:NUM_VALID_DATA
                x = valid_input(idx, :);
                t_lbl = valid_target_lbl(idx, :);
                y = softmax(logsig(x * hidden_matrix + hidden_bias) * output_matrix + output_bias);
                [~, y_lbl] = max(y);
                y_lbls(idx, :) = y_lbl;
                if y_lbl == t_lbl
                    num_correct = num_correct + 1;
                end
            end
            acc = (num_correct) / NUM_VALID_DATA;
            acc_per_epoch(epoch) = acc;
            fprintf('epoch %.0f:  ACC = %.6f\n', epoch, acc);
        end
    end

    % display training process
    fprintf('\nTotal number of epochs: %g\n', NUM_EPOCH);
    fprintf('Final loss: %g\n', loss_per_epoch(NUM_EPOCH));
    plot(1:NUM_EPOCH, loss_per_epoch(1:NUM_EPOCH));
    if VALID
        plot(1:NUM_EPOCH, loss_per_epoch(1:NUM_EPOCH), ...
            1:NUM_EPOCH, acc_per_epoch(1:NUM_EPOCH));
    end
    legend('Training');
    ylabel('Loss');xlabel('Epoch');
    % plot(1:NUM_EPOCH*NUM_TRAIN_DATA, loss_per_step(1:NUM_EPOCH*NUM_TRAIN_DATA));
    % legend('Training');
    % ylabel('Loss');xlabel('Step');

    % Testing
    num_correct=0;
    y_lbls = zeros(NUM_TEST_DATA, 1);
    for idx = 1:NUM_TEST_DATA
        x = test_data(idx, :);
        t_lbl = target_lbl(idx, :);
        % y_ = target_onehot(idx, :);
        y = softmax(logsig(x * hidden_matrix + hidden_bias) * output_matrix + output_bias);
        [~, y_lbl] = max(y);
        y_lbls(idx, :) = y_lbl;
        if y_lbl == t_lbl
            num_correct = num_correct + 1;
        end
    end
    % disp(y_lbls');
    Tot_Percent = (num_correct) / NUM_TEST_DATA;
    fprintf('Test correct percent: %f\n', Tot_Percent);

    % save variables
    if SAVE
        file_name = "ORL_final_"+Tot_Percent*1000+"_"+LR+".mat";
        save(file_name, ...
            "train_data", "test_data", "target_lbl", ...
            "LR", "NUM_EPOCH", ...
            "hidden_matrix", "hidden_bias", ...
            "output_matrix", "output_bias", ...
            '-mat')
    end
end

function [ce] = cross_entropy(labels , logits)
    q = -log(logits);
    p = labels;
    ce = p .* q;
    ce = sum(ce);
end

function [out_vec]=softmax(in_vec)
    exp_in_vec=exp(in_vec);
    out_vec=exp_in_vec/sum(exp_in_vec);
end