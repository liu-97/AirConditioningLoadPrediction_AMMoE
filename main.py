import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import holidays
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False   

warnings.filterwarnings("ignore", category=FutureWarning)

def load_and_preprocess_data(data_path):
    data = pd.read_excel(data_path)
    data['监测时间'] = pd.to_datetime(data['监测时间'], format='%Y/%m/%d %H:%M:%S')
    data['小时'] = data['监测时间'].dt.hour
    data['星期几'] = data['监测时间'].dt.weekday
    data['月份'] = data['监测时间'].dt.month
    data['年份'] = data['监测时间'].dt.year
    data['节假日'] = data['监测时间'].dt.date.isin(holidays.China())
    data['季度'] = data['监测时间'].dt.quarter
    data['是否周末'] = data['星期几'].apply(lambda x: 1 if x >= 5 else 0)
    data['小时_节假日'] = data['小时'] * data['节假日'].astype(int)

    train_list, test_list = [], []
    room_col = "房间"

    for room, group in data.groupby(room_col):
        group = group.sample(frac=1, random_state=42)
        split_idx = int(len(group) * 0.8)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

    train_data = pd.concat(train_list)
    test_data = pd.concat(test_list)

    feature_cols = ['房间'] + train_data.columns[1:11].tolist()
    categorical_cols = train_data[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    X_train = pd.get_dummies(train_data[feature_cols], columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(test_data[feature_cols], columns=categorical_cols, drop_first=True)

    X_train = pd.concat([X_train, train_data[['小时', '星期几', '月份', '年份', '节假日', '季度', '是否周末', '小时_节假日']]], axis=1)
    X_test = pd.concat([X_test, test_data[['小时', '星期几', '月份', '年份', '节假日', '季度', '是否周末', '小时_节假日']]], axis=1)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    y_train = train_data['actual air volume'].astype(float)
    y_test = test_data['actual air volume'].astype(float)

    return X_train, X_test, y_train, y_test

def build_bp_nn(input_dim):
    model = Sequential([
        Dense(1024, input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        ELU(),
        Dropout(0.2),
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        ELU(),
        Dropout(0.2),
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        ELU(),
        Dense(128, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        ELU(),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def auto_algorithm_selection(X_train, y_train, X_test, y_test):
    models = {
        'AdaBoost': AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=6),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'DecisionTree': DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'KNN': KNeighborsRegressor(
            n_neighbors=5,
            metric='minkowski'
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'SVR': SVR(
            C=10,
            gamma='scale',
            epsilon=0.1
        ),
        'GBM': GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        
        results.append({
            'model_name': name,
            'model': model,
            'train_r2': train_r2
        })

    results.sort(key=lambda x: x['train_r2'], reverse=True)
    top_3_models = results[:3]
    
    best_models = [model_info['model'] for model_info in top_3_models]
    
    print("被选中进行集成的三个最佳模型:")
    for i, model_info in enumerate(top_3_models, 1):
        print(f"{i}. {model_info['model_name']}")
    
    return best_models, top_3_models

def visualize_results(y_true, y_pred, selected_models):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label="实际值", color="#F5B3A5", linewidth=1)
    plt.plot(y_pred, label="预测值", color="#AED594", linewidth=1, alpha=0.8)
    
    model_names = [model_info['model_name'] for model_info in selected_models]
    plt.title(f"实际值与预测值对比")
    
    plt.xlabel("样本")
    plt.ylabel("风量")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    data_path = r'E:\学习\投稿\空调负荷预测\SJ.xlsx'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_models, selected_models_info = auto_algorithm_selection(X_train_scaled, y_train, X_test_scaled, y_test)

    X_train_combined = X_train_scaled.copy()
    X_test_combined = X_test_scaled.copy()
    
    for model in best_models:
        train_pred = model.predict(X_train_scaled).reshape(-1, 1)
        test_pred = model.predict(X_test_scaled).reshape(-1, 1)
        X_train_combined = np.hstack((X_train_combined, train_pred))
        X_test_combined = np.hstack((X_test_combined, test_pred))

    bp_nn_model = build_bp_nn(X_train_combined.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_combined, y_train, test_size=0.2, random_state=42
    )
    
    bp_nn_model.fit(X_train_final, y_train_final, epochs=1500, batch_size=32, 
                   validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)

    y_pred = bp_nn_model.predict(X_test_combined, verbose=0).flatten()

    NMBE = 100 * (np.sum(y_test - y_pred) / (len(y_test) * np.mean(y_test)))
    CVRMSE = 100 * (np.sqrt(np.mean((y_test - y_pred) ** 2)) / np.mean(y_test))
    RSQUARED = r2_score(y_test, y_pred)

    print("NMBE: {:.2f}%".format(NMBE))
    print("CVRMSE: {:.2f}%".format(CVRMSE))
    print("R²: {:.4f}".format(RSQUARED))
    
    visualize_results(y_test, y_pred, selected_models_info)

if __name__ == "__main__":
    main()