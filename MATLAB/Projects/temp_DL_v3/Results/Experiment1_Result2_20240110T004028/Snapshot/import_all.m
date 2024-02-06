function [new_train_ds,new_val_ds,new_test_ds, new_combined_ds] = import_all(train_csv, val_csv, test_csv, combined_csv)
    train_tbl = readtable(train_csv);
    train_ds = arrayDatastore(train_tbl,"OutputType","same");
    [numTrain, ~] = size(read(train_ds));
    new_train_ds = transform(train_ds, @(x) [cellfun(@transpose,mat2cell(x{:,1:end-1},ones(1,numTrain)),'UniformOutput',false) , mat2cell(x{:,end},ones(1,numTrain))]);

    val_tbl = readtable(val_csv);
    val_ds = arrayDatastore(val_tbl,"OutputType","same");
    [numVal, ~] = size(read(val_ds));
    new_val_ds = transform(val_ds, @(x) [cellfun(@transpose,mat2cell(x{:,1:end-1},ones(1,numVal)),'UniformOutput',false) , mat2cell(x{:,end},ones(1,numVal))]);

    test_tbl = readtable(test_csv);
    test_ds = arrayDatastore(test_tbl,"OutputType","same");
    [numTest, ~] = size(read(test_ds));
    new_test_ds = transform(test_ds, @(x) [cellfun(@transpose,mat2cell(x{:,1:end-1},ones(1,numTest)),'UniformOutput',false) , mat2cell(x{:,end},ones(1,numTest))]);

    combined_tbl = readtable(combined_csv);
    combined_ds = arrayDatastore(combined_tbl,"OutputType","same");
    [numComb, ~] = size(read(combined_ds));
    new_combined_ds = transform(combined_ds, @(x) [cellfun(@transpose,mat2cell(x{:,1:end-1},ones(1,numComb)),'UniformOutput',false) , mat2cell(x{:,end},ones(1,numComb))]);
end