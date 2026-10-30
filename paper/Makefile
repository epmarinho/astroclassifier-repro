# Astronomy and Computing submission build (recipes require TABs)
TEX      := cnn_transformer_article_ascom
PDF      := $(TEX).pdf
BIB      := cnn_transformer_refs.bib
ELSEVIER := elsevier/elsarticle-3.5
LATEXMK  := latexmk
FLAGS    := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf ascom production clean distclean zip

all: ascom

pdf: ascom

ascom:         ## Build the Astronomy and Computing manuscript
	TEXINPUTS=$(ELSEVIER): BSTINPUTS=$(ELSEVIER): $(LATEXMK) $(FLAGS) $(TEX).tex

production:    ## Build with an explicit pdfLaTeX engine
	TEXINPUTS=$(ELSEVIER): BSTINPUTS=$(ELSEVIER): $(LATEXMK) $(FLAGS) -pdflatex="pdflatex %O %S" $(TEX).tex

clean:         ## Remove auxiliary build files
	$(LATEXMK) -c $(TEX).tex

distclean: clean  ## Also remove generated submission artifacts
	rm -f $(PDF) submission.zip

zip: ascom     ## Build the self-contained submission archive
	rm -f submission.zip
	zip -r submission.zip $(TEX).tex $(PDF) $(BIB) figures_ascom make_ascom_results_figures.py .latexmkrc $(ELSEVIER) README.md Makefile
