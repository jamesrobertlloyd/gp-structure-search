% Turns all figures in a directory into pdfs.
%
% Should be run from source/matlab/
%
% David Duvenaud
% Feb 2013


topdir = '../../figures/decomposition';
latexdir = '../../latex/figures/decomposition';
dirnames = dir(topdir);
isub = [dirnames(:).isdir]; %# returns logical vector
dirnames = {dirnames(isub).name}';
dirnames(ismember(dirnames,{'.','..'})) = [];

dirnames = [];
dirnames{end+1} = '11-Feb-02-solar-s';
dirnames{end+1} = '11-Feb-03-mauna2003-s';
dirnames{end+1} = '31-Jan-v301-airline-months';

dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_0';
dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_1';
dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_2';
dirnames{end+1} = '11-Feb-v4-03-mauna2003-s_max_level_3';

for i = 1:length(dirnames)
    dirname = dirnames{i};
    files = dir([topdir, '/', dirname, '/*.fig']);
    for f_ix = 1:numel(files)
        curfile = [topdir, '/', dirname, '/', files(f_ix).name];
        h = open(curfile);
        outfile = [topdir, '/', dirname, '/', files(f_ix).name];
        pdfname = strrep(outfile, '.fig', '')
        %save2pdf( pdfname, gcf, 600, true );
        export_fig(pdfname, '-pdf');
        close all
    end
end
