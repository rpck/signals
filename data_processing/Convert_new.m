%Quick and perhaps non-optimial way to convert .dat files to .csv files containing real/imaginary values in rows

%Director of .dat files (complex binary data)
directory = '../new_datafiles/';

%Directory for new converted .csv files
newdir = '../test_dataset/';

%Variables for all of the .dat files; we had files called BPSK0.dat, BPSK10.dat, etc. for each SNR. This will gather all of a specified SNR.
BPSK = dir(strcat(directory, 'BPSK*.dat'));
QPSK = dir(strcat(directory, 'QPSK*.dat'));
QAM16 = dir(strcat(directory, 'QAM16*.dat'));
QAM64 = dir(strcat(directory, 'QAM64*.dat'));
VT = dir(strcat(directory, 'VT*.dat'));

%Creates a list of SNR variables above to parse
list = [BPSK QPSK QAM16 QAM64 VT];

%How many datapoints each section will contain (for input)
sections = 500;

%Iterate through list and lay out data as two rows for each input, broken up into groups of 500.
[m, n] = size(list);
for i = 1:m
    disp(strcat('Converting: ', regexprep(list(1, i).name, '_(\d+)', '')));
    currRow = 1;
    output = zeros(2000000 * 5 / sections, sections);

    for j = 1:n 
        %Call matlab file to convert complex binary to real/imag values
        mod = read_complex_binary(strcat(directory, list(j, i).name));
        
        
        real_data = transpose(real(mod));
        imag_data = transpose(imag(mod));
        num = 1;
        %Breaks real/imag data (in proper format) into sections and add to output matrix to ouput on each iteration
        for k = 1:(length(mod) / sections)
            real_section = real_data(num:num + sections - 1);
            imag_section = imag_data(num:num + sections - 1);

            temp = reshape([real_section(:) imag_section(:)]',2*size(real_section,1), []);
            output(currRow:currRow + 1, :) = temp;
            currRow = currRow + 2;
            num = num + sections;
        end
    end
    csvwrite(strcat(newdir, strrep(regexprep(list(1, i).name, '_(\d+)', ''), '.dat', '.csv')), output);
end
disp('Finished conversion!');
