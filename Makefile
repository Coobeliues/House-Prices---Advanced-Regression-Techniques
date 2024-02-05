
lint:
	isort .
	flake8 
	black

submissions:
	kaggle competitions submissions -c house-prices-advanced-regression-techniques

submit:
	kaggle competitions submit -c house-prices-advanced-regression-techniques -f $(FILES) -m "$(COMMENT)"