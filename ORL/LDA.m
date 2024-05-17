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
