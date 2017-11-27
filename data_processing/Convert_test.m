directory = '../new_datafiles/';
newdir = '../test_dataset/';

BPSK = dir(strcat(directory, 'BPSK*.dat'));
QPSK = dir(strcat(directory, 'QPSK*.dat'));
QAM16 = dir(strcat(directory, 'QAM16*.dat'));
QAM64 = dir(strcat(directory, 'QAM64*.dat'));
VT = dir(strcat(directory, 'VT*.dat'));

list = [BPSK QPSK QAM16 QAM64 VT];
sections = 500;

[m, n] = size(list);
for i = 1:m
    disp(strcat('Converting: ', regexprep(list(1, i).name, '_(\d+)', '')));
    currRow = 1;
    output = zeros(2000000 * 5 / sections, sections);

    for j = 1:n 
      var = regexprep(list(j, i).name, '\w+_', '');
      if (strcmp(var, '0.dat') ~= 0)

      mod = read_complex_binary(strcat(directory, list(j, i).name));

        real_data = transpose(real(mod));
        imag_data = transpose(imag(mod));
        num = 1;
        for k = 1:(length(mod) / sections)
            real_section = real_data(num:num + sections - 1);
            imag_section = imag_data(num:num + sections - 1);

            temp = reshape([real_section(:) imag_section(:)]',2*size(real_section,1), []);
            output(currRow:currRow + 1, :) = temp;
            currRow = currRow + 2;
            num = num + sections;
        end
      end
    end
    csvwrite(strcat(newdir, strrep(regexprep(list(1, i).name, '_(\d+)', ''), '.dat', '.csv')), output);
end
disp('Finished conversion!');
