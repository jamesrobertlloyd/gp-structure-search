% Turns all figures in a directory into pdfs.
%
% Should be run from source/matlab/
%
% David Duvenaud
% Feb 2013


topdir = '../../figures/decomposition/';
dirnames = dir(topdir);
isub = [dirnames(:).isdir]; %# returns logical vector
dirnames = {dirnames(isub).name}';
dirnames(ismember(dirnames,{'.','..'})) = [];

dirnames = [];
%dirnames{1} = '../../figures/decomposition/11-Feb-02-solar-s';

%dirnames{2} = '../../figures/decomposition/11-Feb-03-mauna2003-s';
dirnames{1} = '../../figures/decomposition/31-Jan-v301-airline-months';

for i = 1:length(dirnames)
    dirname = dirnames{i};
    files = dir([topdir, '/', dirname, '/*.fig']);
    for f_ix = 1:numel(files)
        curfile = [topdir, '/', dirname, '/', files(f_ix).name];
        h = open(curfile);
        pdfname = strrep(curfile, '.fig', '.pdf')
        save2pdf( pdfname, gcf, 600, true );
        close all
    end
end
