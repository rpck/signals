directory = '../dataset/';
newdir = '../dataset/';

list = dir(strcat(directory, '*.dat'));
sections = 512;

for i = 1:length(list)
    disp(strcat('Converting: ', list(i).name));
    
    mod = read_complex_binary(strcat(directory, list(i).name));
    
    
    real_data = transpose(real(mod));
    imag_data = transpose(imag(mod));
    num = 1;
    output = zeros(length(mod) / sections, sections);
    currRow = 1;
    for j = 1:(length(mod) / sections)
        real_section = real_data(num:num + sections - 1);
        imag_section = imag_data(num:num + sections - 1);

        temp = reshape([real_section(:) imag_section(:)]',2*size(real_section,1), []);
        output(currRow:currRow + 1, :) = temp;
        currRow = currRow + 2;
        num = num + sections;
    end
    csvwrite(strcat(newdir, strrep(list(i).name, '.dat', '.csv')), output);

end
disp('Finished conversion!');
