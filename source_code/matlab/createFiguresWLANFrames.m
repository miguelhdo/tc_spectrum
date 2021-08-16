function createFiguresWLANFrames(json_file)
    %'best_results.json'
    [~,name,~] = fileparts(json_file);
    fname = json_file;
    fid = fopen(fname);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);
    
    %X and Ys for accuracy figure
    
    X1 = double(string(val.LengthIQSamples));
    YMatrix = [double(string(val.GRU.BestTestAccuracy)), double(string(val.CNN.BestTestAccuracy))];
    createFigureBestTestAccuracy(X1,YMatrix, name);
    
    %X and Ys mean time per epoch
    YMatrix2 = [double(string(val.GRU.MeanTimePerEpoch)), double(string(val.CNN.MeanTimePerEpoch))];
    createFigureTrainingTime(X1,YMatrix2, name)
end