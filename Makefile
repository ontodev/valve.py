## Grammar
#
# Generate grammar, then ...
# 1. Remove init babel from first line
# 2. Encase grammar in triple quotes to allow for line breaks
# 3. Replace literal '\n' with line breaks
# 4. Fix extra escaping
# 5. Fix extra escaping on single quotes
# 6. Add escaping for double quotes (double quotes must be escaped within double quoted text)
# 7. Fix empty escaping (Lark will yell at you for \\)
# 8. Add x flag to regex using '\n'
# 9. Get the first result and convert to a Python dictionary
# 10. Format using black

build build/distinct:
	mkdir -p $@

build/valve_grammar.ne: | build
	curl -Lk https://raw.githubusercontent.com/ontodev/valve.js/main/valve_grammar.ne > $@

build/nearley: | build
	cd build && git clone https://github.com/Hardmath123/nearley

build/valve_grammar.py: build/valve_grammar.ne | build/nearley
	python3 -m lark.tools.nearley $< start $| --es6 > $@

valve/parse.py: build/valve_grammar.py
	tail -n +2 $< | \
	perl -pe "s/grammar = (.+)/grammar = ''\1''/g" | \
	perl -pe 's/(?<!\\)\\n/\n/gx' | \
	perl -pe 's/\\\\/\\/gx' | \
	perl -pe "s/\\\'/'/g" | \
	perl -pe 's/"\\\"/"\\\\"/g' | \
	perl -pe 's/"\\\\"$$/"\\\\\\\\"/g' | \
	perl -pe 's/(\/\[.*\\n.*]\/)/\1x/g'| \
	perl -pe 's/\[\^\/\]/[^\\\/]/g' | \
	sed -e "s/parse(text))/parse(text)).to_dict()/g" > $@
	black --line-length 100 $@


## Linting

.PHONY: lint
lint:
	flake8 --max-line-length 100 --ignore E203,W503 $(PYTHON_FILES)
	black --line-length 100 --quiet --check $(PYTHON_FILES)

.PHONY: format
format:
	black --line-length 100 $(PYTHON_FILES)


## Testing

valve-main:
	git clone https://github.com/ontodev/valve.git $@

build/errors.tsv: valve-main | build
	valve valve-main/tests/inputs -o $@ || true

build/errors-distinct.tsv: valve-main | build/distinct
	valve valve-main/tests/inputs -d build/distinct -o $@ || true

python-diff: valve-main build/errors.tsv
	python3 valve-main/tests/compare.py valve-main/tests/errors.tsv build/errors.tsv

python-diff-distinct: valve-main build/errors-distinct.tsv
	python3 valve-main/tests/compare.py valve-main/tests/errors-distinct.tsv build/errors-distinct.tsv

.PHONY: unit-test
unit-test:
	pytest tests

.PHONY: integration-test
integration-test:
	make python-diff
	make python-diff-distinct
	@echo "SUCCESS: Tests ran as expected."

.PHONY: test
test: unit-test integration-test

.PHONY: clean
clean:
	rm -rf build valve-main
