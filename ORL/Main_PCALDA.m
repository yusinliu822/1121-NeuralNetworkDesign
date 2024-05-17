function Main_PCALDA
    % Midterm Project: read ORL data and do PCA and LDA
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