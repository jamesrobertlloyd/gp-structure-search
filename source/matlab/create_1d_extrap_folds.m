directory = '../../data/1d_data/';
fold_directory = '../../data/1d_extrap_folds/';
folds = 10;
file_list = dir(directory);

for i = 1:length(file_list)
    a_file = file_list(i);
    if ~a_file.isdir
        load([directory a_file.name]);
        X_total = X;
        y_total = y;
        for fold = 1:(folds-1)
            % Create folds of various lengths - assume 1d ordered data
            X = X_total(1:floor(end*fold/folds));
            y = y_total(1:floor(end*fold/folds));
            Xtest = X_total(floor(end*fold/folds):end);
            ytest = y_total(floor(end*fold/folds):end);
            [~, data_name, ~] = fileparts(a_file.name);
            output_name = [fold_directory data_name '-ex-fold-' ...
                           int2str(fold) 'of' int2str(folds) '.mat'];
            save( output_name, 'X', 'y', 'Xtest', 'ytest' );
        end
    end
end
