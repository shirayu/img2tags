
all: lint_node lint_python

TARGET_DIRS:=./img2tags
OUTPUT_STAT:=/dev/stdout

ruff:
	ruff format --respect-gitignore --check
	ruff --respect-gitignore

yamllint:
	find . \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.yml' -print \
		| xargs yamllint --no-warnings

lint_python: ruff yamllint


pyright:
	npx pyright

markdownlint:
	find . -type d \( -name node_modules -o -name .venv \) -prune -o -type f -name '*.md' -print \
	| xargs npx markdownlint --config ./.markdownlint.json

lint_node: markdownlint pyright

