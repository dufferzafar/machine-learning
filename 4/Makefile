C=\033[1;035m # Color
NC=\033[0m # No Color

default: run

run:
	@echo "Run"

download:
	@kaggle competitions download -c col-774-spring-2018

submit:
	@printf "$(C)Uploading file '$(f)' with message '$(m)'$(NC)\n"
	@kaggle competitions submit -c col-774-spring-2018 -f "$(f)" -m "$(m)"

kaggle:
	@xdg-open "https://www.kaggle.com/c/col-774-spring-2018"
