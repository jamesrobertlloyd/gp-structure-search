% Turns all figures in a directory into pdfs.

topdir = '../../figures/decomposition/';
dirnames = dir(topdir);
isub = [dirnames(:).isdir]; %# returns logical vector
dirnames = {dirnames(isub).name}';
dirnames(ismember(dirnames,{'.','..'})) = [];

%dirnames = {'../../figures/decomposition/03-mauna2003_max_level_8'};

for i = 1:length(dirnames)
    dirname = dirnames{i};
    files = dir([topdir, '/', dirname, '/*.fig']);
    for f_ix = 1:numel(files)
        curfile = [topdir, '/', dirname, '/', files(f_ix).name];
        h = open(curfile);
        pdfname = strrep(curfile, '.fig', '.pdf')
        save2pdf( pdfname, gcf, 600, true );
        %close all
    end
end
