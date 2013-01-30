% Turns all figures in a directory into pdfs.

dirname = '/homes/mlghomes/dkd23/git/gp-structure-search/figures/decomposition/01-airline/';

files = dir([dirname '*.fig']);
for f_ix = 1:numel(files)
    curfile = [dirname, files(f_ix).name];
    h = open(curfile);
    pdfname = strrep(curfile, '.fig', '.pdf')
    save2pdf( pdfname, gcf, 600, true );
end
