default: run

run:
	@echo "Run"

download:
	@kaggle competitions download -c col-774-spring-2018

submit:
	@# TODO: Ensure that older CSVs get backed up?
	@kaggle competitions submit -c col-774-spring-2018 -f "kaggle_submit.csv" -m "Tried this? & That?"
