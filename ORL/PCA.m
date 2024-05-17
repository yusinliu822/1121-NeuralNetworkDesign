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
