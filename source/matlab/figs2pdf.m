% Turns all figures in a directory into pdfs.

dirname = '../../figures/decomposition/02-solar/';

files = dir([dirname '*.fig']);
for f_ix = 1:numel(files)
    curfile = [dirname, files(f_ix).name];
    h = open(curfile);
    pdfname = strrep(curfile, '.fig', '.pdf')
    save2pdf( pdfname, gcf, 600, true );
end
