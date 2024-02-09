
lint:
	isort .
	flake8 
	black

submissions:
	kaggle competitions submissions -c house-prices-advanced-regression-techniques

submit:
	kaggle competitions submit -c house-prices-advanced-regression-techniques -f $(FILES) -m "$(COMMENT)"

# по идее можно было бы сделать все это через мжйкфайл, запуск кода, подбор методов предобработки, модельки с параметрами и тд, но чет думаю это слишком