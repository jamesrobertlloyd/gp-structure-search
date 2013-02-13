function set_fig_units_cm( width, height )

set(gcf,'units','centimeters');
%set(gcf, 'ActivePositionProperty', 'tightinset');
pos = get(gcf,'position');
set(gcf,'position',[pos(1), pos(2),width,height]);

% If cutting off text, do this
apos = get(gca,'position');
set(gca,'position',[apos(1), apos(2) * 1.3, apos(3), apos(4) * 0.9]);
