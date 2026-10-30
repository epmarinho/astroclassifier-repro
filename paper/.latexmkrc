# Shared latexmk configuration for the Astronomy and Computing manuscript.
# LaTeX Workshop runs latexmk with paper/ as its working directory.
$max_repeat = 10;
$ENV{'TEXINPUTS'} = './elsevier/elsarticle-3.5:' . ($ENV{'TEXINPUTS'} || '');
$ENV{'BSTINPUTS'} = './elsevier/elsarticle-3.5:' . ($ENV{'BSTINPUTS'} || '');
