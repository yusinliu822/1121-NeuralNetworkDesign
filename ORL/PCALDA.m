function PCALDA
    % Midterm Project: read ORL data and do PCA and LDA
    % One file version concat all functions.
    %{
        ORL dataset is a dataset of 400 pictures from 40 people,
        each person took different 10 pictures, mark as 1 to 10.
        In ORL_RawData.mat, 400 pictures had been split to training set and
        testing set, one person's 5 pictures marked with odd numbers are in
        training set and the even ones are in testing set.
    %}
    
    PCA_DIM = 80;
    LDA_DIM = 30;
    NUM_CLASS = 40;
    NUM_SAMPLE = 5;

    data = load("ORL_RawData.mat");
    train_data = data.ORLrawdataTrain;
    test_data = data.ORLrawdataTest;
    
    origin_mean_data = mean(train_data);
    [pca_data, pca_project] = PCA(train_data, PCA_DIM);
    [lda_data, lca_project] = LDA(pca_data, LDA_DIM, NUM_CLASS);

    PCALDA_eval(test_data, lda_data, origin_mean_data, pca_project, lca_project, NUM_CLASS, NUM_SAMPLE)
end

function [pca_data, pca_project] = PCA(data, out_dim)
    % PCA
    mean_data = mean(data);
    zero_data = data - mean_data;

    cov_mat = zero_data' * zero_data;

    [eig_vec, eig_val] = eigs(cov_mat, out_dim);
    [~, sort_idx] = sort(diag(eig_val), 'descend');
    pca_project = eig_vec(:, sort_idx);
    
    pca_data = zero_data * pca_project;
end

function [lda_data, lca_project] = LDA(data, out_dim, num_class)
    % LDA
    
    % n: number of total data, m: num of features
    % n = size(data, 1);
    m = size(data, 2);
    
    % Calculate the within-class scatter matrix
    within_scatter = zeros(m);
    for i = 1:num_class
        class_data = data(5*i-4:5*i, :);
        within_scatter = within_scatter + class_data' * class_data;
    end

    % Calculate the between-class scatter matrix
    mean_classes = zeros(num_class, m);
    for i = 1:num_class
        class_data = data(5*i-4:5*i, :);
        mean_classes(i, :) = mean(class_data);
    end
    between_scatter = mean_classes' * mean_classes;
    
    % Solve the generalized eigenvalue problem
    matrix = pinv(within_scatter)*between_scatter;
    [eig_vec, eig_val] = eigs(matrix, out_dim);
    [~, sort_idx] = sort(diag(eig_val), 'descend');
    lca_project = eig_vec(:, sort_idx);

    % Transfer data
    lda_data = data * lca_project;
end

function PCALDA_eval(test_data, train_data, origin_mean_data, pca_project, lca_project, num_class, num_sample)
    
    % Zero-Mean
    zero_mean_data = test_data - origin_mean_data;

    % PCALDA
    test_data = zero_mean_data * pca_project * lca_project;
    
    num_correct = 0;

    for k = 1:num_class
        for r = 1:num_sample
            cur_test = test_data(num_sample*(k-1)+r, :);
            predict_lbl = predict(cur_test, train_data, num_class, num_sample);
            if predict_lbl == k
                num_correct = num_correct + 1;
            end
        end
    end

    disp("Acccurcy: ")
    disp(num_correct./200)

end

function [predict_label] = predict(test_d, train_data, num_class, num_sample)
    min_dist = Inf;
    predict_label = -1;
    for i = 1:num_class
        for j = 1:num_sample
            cur_data = train_data(num_sample*(i-1)+j, :);
            diff = cur_data - test_d;
            dist = diff * diff';
            if dist < min_dist
                min_dist = dist;
                predict_label = i;
            end
        end
    end
end
