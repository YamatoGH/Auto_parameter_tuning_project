import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
from generate_LLM_response import get_response_from_llm





# json形式のテキストを各パラメータの辞書と継続状態のブール値で返す
def get_parameters_from_json(json_response):
    response_dict = json.loads(json_response)  # jsonテキストとして辞書形式に読み込み
    params_range_list_dict = {}                          # 初期化

    for key, value in response_dict.items():
        params_range_list_dict[key] = eval(value).tolist()

    return params_range_list_dict




system_message = "You are an assistant for tuning parameters of LightGBM."




# 自動チューニングを実行　　返り値はデータフレーム
# X_train  -> LGBMRegressor()実行に適した訓練データのX
# y_train  -> LGBMRegressor()実行に適した訓練データのy
# params_range_list_dict  ->  gridsearchを行うための辞書   例){'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 150, 200], 'max_depth': [5, 20]}
# max_try_count  ->  最大探索実行回数
# early_stopping  ->  早期停止
def run_auto_tuning_lgbm(X_train, y_train, params_range_list_dict, max_try_count, early_stopping=5):
    keep_state = True  
    lgbm = LGBMRegressor(random_state=42)  # モデル初期化
    try_count = 0                           # パラメータ探索の実行回数
    hist_parameters_grid = []
    max_try_count = max_try_count

    not_improve_count = 0
    rmse_list = []
    model_list = []

    while keep_state:
        print(f"{try_count}th trial")
        # ハイパーパラメータの設定範囲
        param_grid = params_range_list_dict    # 引数

        # GridSearchCVの設定
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  # 評価指標：MSE (負の値で返される)
            cv=3,  
            verbose=0,
            n_jobs=-1   # 並列実行の指定（全プロセス使用）
            )
        
        grid_search.fit(X_train, y_train)
        lgbm_model = grid_search.best_estimator_
        X_pred = lgbm_model.predict(X_train)
        new_rmse = mean_squared_error(X_pred,y_train)

        if try_count>1:
            min_rmse = min(rmse_list)
            improvement_rate = (min_rmse - new_rmse) / min_rmse

            # 0.1%の改善を基準に早期停止処理
            if improvement_rate <= 0.001:
                not_improve_count += 1

            else:
                not_improve_count = 0
            
            if not_improve_count >=early_stopping:
                keep_state = False
            

        model_list.append(lgbm_model)
        rmse_list.append(new_rmse)


        print()
        current_best_parameters = f"""
        Best parameters: {grid_search.best_params_}
        RMSE: {new_rmse}
        """

        hist_parameters_grid.append(param_grid)
        print(hist_parameters_grid)
        print(current_best_parameters)
        print()
        
        # チューニングプロンプト
        tuning_msg = f"""
            Tune the parameters of LightGBM following below.

            I give you parameters-grid history of past trials and current the best LightGBM parameters with a metric.
            Pay attention that vacant space for parameters-grid history means "this is first trial so that there is no parameters history".

            parameters-grid history: {hist_parameters_grid}
            current best parameters: {current_best_parameters}

            Suggest new parameters-grid BY USING "numpy arange" and output it as "JSON format" by refering parameters-grid history and current best parameters with the metric.
            Only tune the parameters included in the current best parameters.
            
            Don't use the same parameter configuration.
            Only output suggested parameters by "JSON format".
            Do not include explanations, comments, or any other text.

            Here is one example of output format:
                "learning_rate": "np.arange(0.1, 0.3, 0.1)",
                "n_estimators": "np.arange(100, 140, 10)",
                "max_depth": "np.arange(10, 15, 1)"


            Again, pay attention to output format especialy in parameters-grid BY USING "numpy arange".
        """
        
        suggested_parameters_grid, new_msg_hist = get_response_from_llm(msg=tuning_msg, 
                                                            system_message=system_message,
                                                            )

        params_range_list_dict = get_parameters_from_json(suggested_parameters_grid)

        try_count += 1
        if try_count==max_try_count:
            keep_state = False

    results_df = pd.DataFrame({
        "model": model_list,
        "RMSE": rmse_list
        })
        

    return results_df
