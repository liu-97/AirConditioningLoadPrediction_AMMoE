# AMMoE Framework: Automated Mixture of Experts for Cross-Building Energy Prediction
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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, ELU, Input, Concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import holidays
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False   
warnings.filterwarnings("ignore", category=FutureWarning)


# ======================= # 1. AMMoE Data Processor # =====================
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

# ============ # 2.The Gating Network is built using a BP-NN # =============
def build_gating_network(input_dim, n_experts):
    inputs = Input(shape=(input_dim,))
    
    x = Dense(256, kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    gate_output = Dense(n_experts, activation='softmax', name='gate_output')(x)
    
    model = Model(inputs=inputs, outputs=gate_output, name='gating_network')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
    
    return model

# ============== # 3.Expert Models # ===============
def train_expert_models(X_train, y_train, n_experts=3):
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
            n_estimators=300,
            learning_rate=0.05,
            min_child_samples=15,
            random_state=42,
            verbose=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
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
        )
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        
        results.append({
            'model_name': name,
            'model': model,
            'train_r2': train_r2
        })
    
    results.sort(key=lambda x: x['train_r2'], reverse=True)
    selected_models = results[:n_experts]
    
    return selected_models

# ============== # 4.Create AMMoE # ===============
def create_amoe_model(gating_network, expert_models, input_dim):
    inputs = Input(shape=(input_dim,))
    gate_weights = gating_network(inputs)
    expert_predictions = []
    for i, expert_info in enumerate(expert_models):
        expert_pred = Lambda(
            lambda x: tf.py_function(
                func=lambda x: expert_info['model'].predict(x.numpy()).astype(np.float32),
                inp=[x],
                Tout=tf.float32
            ),
            output_shape=(1,),
            name=f'expert_{i}_prediction'
        )(inputs)
        expert_predictions.append(expert_pred)
    
    expert_stack = Concatenate(axis=-1, name='expert_stack')(expert_predictions)
    
    gate_weights_expanded = tf.expand_dims(gate_weights, axis=-1)  
    expert_stack_expanded = tf.expand_dims(expert_stack, axis=-1)  
    
    weighted_predictions = tf.reduce_sum(
        gate_weights_expanded * expert_stack_expanded, 
        axis=1
    )  
    
    final_output = tf.squeeze(weighted_predictions, axis=-1, name='final_output')
    
    model = Model(inputs=inputs, outputs=final_output, name='AMMoE')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

def train_amoe_model(amoe_model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32):
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=50, 
        restore_best_weights=True, 
        verbose=1
    )
    
    history = amoe_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def visualize_amoe_results(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label="actual value", color="#F5B3A5", linewidth=1)
    plt.plot(y_pred, label="predicted value", color="#AED594", linewidth=1, alpha=0.8)
    plt.title("Comparison of actual value and predicted value")
    plt.xlabel("sample")
    plt.ylabel("Air volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main_amoe():
    data_path = r'data.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    expert_models = train_expert_models(X_train_scaled, y_train, n_experts=3)
    
    gating_network = build_gating_network(X_train_scaled.shape[1], len(expert_models))
    
    expert_predictions_train = []
    for expert in expert_models:
        pred = expert['model'].predict(X_train_scaled)
        pred = np.maximum(pred, 0) 
        expert_predictions_train.append(pred)
    
    expert_predictions_train = np.array(expert_predictions_train).T  # [n_samples, n_experts]
    
    errors_per_expert = (expert_predictions_train - y_train.values.reshape(-1, 1)) ** 2
    best_expert_indices = np.argmin(errors_per_expert, axis=1)
    
    y_gate_train = np.eye(len(expert_models))[best_expert_indices]
    
    X_train_gate, X_val_gate, y_train_gate, y_val_gate = train_test_split(
        X_train_scaled, y_gate_train, test_size=0.2, random_state=42
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    gate_history = gating_network.fit(
        X_train_gate, y_train_gate,
        epochs=500,
        batch_size=32,
        validation_data=(X_val_gate, y_val_gate),
        callbacks=[early_stop],
        verbose=1
    )
    
    gate_weights_test = gating_network.predict(X_test_scaled)
    
    expert_predictions_test = []
    for expert in expert_models:
        pred = expert['model'].predict(X_test_scaled)
        pred = np.maximum(pred, 0)
        expert_predictions_test.append(pred)
    
    expert_predictions_test = np.array(expert_predictions_test).T  # [n_samples, n_experts]
    
    y_pred = np.sum(gate_weights_test * expert_predictions_test, axis=1)
    y_pred = np.maximum(y_pred, 0)
    
 # ============== # 5.Evaluate AMMoE # ===============
    NMBE = 100 * (np.sum(y_test - y_pred) / (len(y_test) * np.mean(y_test)))
    CVRMSE = 100 * (np.sqrt(np.mean((y_test - y_pred) ** 2)) / np.mean(y_test))
    RSQUARED = r2_score(y_test, y_pred)
    
    print("\n=== AMMoE model evaluation results ===")
    print("NMBE: {:.2f}%".format(NMBE))
    print("CVRMSE: {:.2f}%".format(CVRMSE))
    print("R²: {:.4f}".format(RSQUARED))
    
    expert_names = [exp['model_name'] for exp in expert_models]
    expert_names_str = "、".join(expert_names)

    print(f"Selected expert models：{expert_names_str}")
    visualize_amoe_results(y_test, y_pred)

if __name__ == "__main__":
    import tensorflow as tf
    main_amoe()
