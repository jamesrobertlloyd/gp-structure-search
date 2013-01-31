% Turns all figures in a directory into pdfs.

dirnames = {'../../figures/decomposition/03-mauna2003/'};

for i = 1:length(dirnames)
    dirname = dirnames{i};
    files = dir([dirname '*.fig']);
    for f_ix = 1:numel(files)
        curfile = [dirname, files(f_ix).name];
        h = open(curfile);
        pdfname = strrep(curfile, '.fig', '.pdf')
        save2pdf( pdfname, gcf, 600, true );
    end
end
