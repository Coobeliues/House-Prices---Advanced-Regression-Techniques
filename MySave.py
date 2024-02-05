# import pandas as pd
# import numpy as np
# import os
# import MyTrain
# import MyModels
# import prepprocessing

# # print(best_params_svr, best_params_knn, sep="\n")

# # # Получение и сохранение предсказаний каждой модели в отдельный CSV файл
# # for model_name, model in models_tuned.items():
# #     predictions = pd.DataFrame({
# #         "Id": test["Id"],
# #         "SalePrice": np.expm1(model.predict(X_test_processed))
# #     })
# #     predictions.to_csv(f"predictions_{model_name}_0130.csv", index=False)


# def saveMyfiles():
#     # Создаем директорию для сохранения предсказаний, если ее еще нет
#     output_directory = "predictions"
#     os.makedirs(output_directory, exist_ok=True)

#     # Создаем текстовый файл для записи информации о методах предобработки
#     info_file_path = os.path.join(output_directory, "models_info.txt")
#     info_file = open(info_file_path, "a")

#     # Счетчик для идентификационных номеров
#     counter = 1

#     # Получение и сохранение предсказаний каждой модели в отдельный CSV файл
#     for model_name, model in models_tuned.items():
#         # Применяем методы предобработки к тестовым данным
#         X_test_processed = preprocessor.fit_transform(X_test)
        
#         # Сохраняем информацию о методах предобработки
#         preprocessing_info = {
#             'imputer_strategy': imputer_strategy,
#             'scaler': scaler,
#             'cat_imputer_strategy': cat_imputer_strategy
#         }

#         # Создаем DataFrame с предсказаниями
#         predictions = pd.DataFrame({
#             "Id": test["Id"],
#             "SalePrice": np.expm1(model.predict(X_test_processed))
#         })

#         # Добавляем идентификационный номер к имени файла
#         file_name = f"pred_{model_name}_{counter:03}.csv"
        
#         # Сохраняем предсказания в CSV файл
#         predictions.to_csv(os.path.join(output_directory, file_name), index=False)

#         # Записываем информацию о методах предобработки в текстовый файл
#         info_file.write(f"Id {counter:03}: Cat Imputer = {preprocessing_info['cat_imputer_strategy']}, Num Imputer = {preprocessing_info['imputer_strategy']}\n")

#         # Увеличиваем счетчик
#         counter += 1

#     # Закрываем текстовый файл
#     info_file.close()

