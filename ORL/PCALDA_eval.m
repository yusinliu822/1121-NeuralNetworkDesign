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
