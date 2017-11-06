directory = '../dataset/';
newdir = '../dataset_csv/';

sections = 1000;

list = dir(strcat(directory, '*.dat'));

for i = 1:length(list)
    disp(strcat('Converting: ', list(i).name));
    
    mod = read_complex_binary(strcat(directory, list(i).name));
    real_data = transpose(real(mod));
    imag_data = transpose(imag(mod));
    num = 1;
    for j = 1:(length(mod) / sections)
        temp = [real_data(num:num + 999); imag_data(num:num + 999)];
        csvwrite(strcat(newdir, int2str(j), '_', strrep(list(i).name, '.dat', '.csv')), temp);
        num = num + 1000;
        ind = ind + 1;
    end
end
disp('Finished conversion!');
