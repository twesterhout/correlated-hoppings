all: pdf
pdf: 01_main.pdf

.PHONY: clean 

.PHONY: dependencies
dependencies:
	tlmgr install revtex textcase pdftexcmds infwarerr

01_main.pdf: 01_main.tex references.bib assets/ferromagnetism.pdf
	pdflatex $<
	pdflatex $<
	bibtex 01_main.aux
	pdflatex $<
	pdflatex $<

arxiv: 01_main.tex 01_main.bbl 02_supplement.tex assets/ferromagnetism.pdf assets/various_sizes.pdf
	mkdir -p submission
	cp -r 01_main.tex 01_main.bbl 02_supplement.tex assets submission
	tar -cf submission.tar submission/
	rm -r submission

clean:
	rm -f paper.pdf paper.tex
	rm -f paper.aux paper.bcf paper.log paper.out paper.run.xml
	rm -rf build/
	rm -rf submission.tar submission/
