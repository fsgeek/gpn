.PHONY: build clean

BASE_FILE := main
BIBTEX_FILE := references.bib
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
OUTPUT := draft-$(BASE_FILE)-$(TIMESTAMP).pdf
LATEXMK_FLAGS := -lualatex --jobname=$(OUTPUT:.pdf=) -interaction=nonstopmode -file-line-error

build:  ${BASE_FILE}.pdf

${BASE_FILE}.pdf : ${BASE_FILE}.tex ${BIBTEX_FILE}
	latexmk $(LATEXMK_FLAGS) $(BASE_FILE).tex
	-mv $(BASE_FILE).pdf $(OUTPUT)
	-ln  $(OUTPUT) $(BASE_FILE).pdf

clean:
	latexmk -C
	rm -rf build
	rm -f draft-$(BASE_FILE)-*.pdf

.DEFAULT_GOAL := build
