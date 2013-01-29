% Turns all figures in a directory into pdfs.

dirname = '/homes/mlghomes/dkd23/git/gp-structure-search/figures/decomposition/';

files = dir([dirname '*.fig']);
for f_ix = 1:numel(files)
    curfile = [dirname, file(f_ix).name];
    h = open(curfile);
    pdfname = strrep(curfile, '.fig', '.pdf')
    save2pdf( pdfname, gcf, 600, true );
end
