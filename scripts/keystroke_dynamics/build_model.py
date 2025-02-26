import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential


def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        # Predict probabilities
        y_pred_prob = model.predict(X_test).flatten()

        # Handle potential NaN or problematic predictions
        if np.isnan(y_pred_prob).any():
            print(f"Warning: NaN values found in predictions for {name}")
            # Replace NaNs with a default probability (e.g., 0.5)
            y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.5)

        # Robust ROC curve calculation
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        except ValueError as e:
            print(f"Error calculating ROC curve for {name}: {e}")
            print("Prediction probabilities:", y_pred_prob)
            print("True labels:", y_test)
            # Skip this model if ROC curve cannot be calculated
            continue

        # Check if ROC curve data is valid
        if len(fpr) == 0 or len(tpr) == 0 or len(thresholds) == 0:
            print(f"Invalid ROC curve data for {name}")
            continue

        # Calculate False Negative Rate (FNR)
        fnr = 1 - tpr

        # Find the threshold where FPR and FNR are closest
        diff = np.abs(fpr - fnr)

        # Ensure diff is not all NaNs
        if np.isnan(diff).all():
            print(f"Warning: Could not find valid EER threshold for {name}")
            # Use a default threshold of 0.5
            eer_threshold = 0.5
            eer = 0.5
        else:
            # Find the index where the difference is minimized
            eer_index = np.nanargmin(diff)
            eer_threshold = thresholds[eer_index]
            eer = (fpr[eer_index] + fnr[eer_index]) / 2

        # Predictions using the EER threshold
        y_pred = (y_pred_prob >= eer_threshold).astype(int)

        # Compute performance metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Compute False Accept Rate (FAR) and False Reject Rate (FRR)
        cm = confusion_matrix(y_test, y_pred)

        # Ensure the confusion matrix is 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (tn + fp) if (tn + fp) > 0 else 0
            frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        else:
            far = frr = 0

        results[name] = {
            "accuracy": accuracy,
            "eer": eer,
            "far": far,
            "frr": frr,
            "threshold": eer_threshold,
        }

    return results


# モデル構築関数
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_cnn_model(input_shape):
    model = Sequential()
    model.add(
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_hybrid_model(input_shape):
    model = Sequential()
    model.add(
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# データの読み込み関数
def load_keystroke_data(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            speed_label = "normal"
            if "fast" in filename:
                speed_label = "fast"
            elif "slow" in filename:
                speed_label = "slow"

            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            # キーと時間値を抽出
            df["key_code"] = df["KEY"].apply(lambda x: x.split("=")[1].split("(")[0])
            df["key_char"] = df["KEY"].apply(
                lambda x: x.split("(")[1].split(")")[0] if "(" in x else ""
            )
            df["timestamp"] = df["TIMESTAMP"].apply(lambda x: int(x.split("=")[1]))

            # HOLDとFLIGHTの値を抽出
            df["value"] = df["TIMING"].apply(lambda x: int(x.split("=")[1]))
            df["timing_type"] = df["TIMING"].apply(lambda x: x.split("=")[0])

            # 速度ラベルを追加
            df["speed_label"] = speed_label
            # ユーザーIDをファイル名から抽出
            user_id = filename.split("_")[0] if "_" in filename else "unknown"
            df["user_id"] = user_id

            all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# 特徴抽出関数
def extract_keystroke_features(df, user_id):
    """キーストロークから特徴を抽出し、ピボットテーブル形式で返す"""
    features_dict = {}

    # HOLD
    hold_times = df[(df["EVENT"] == "UP") & (df["timing_type"] == "HOLD")]
    for key in df["key_code"].unique():
        key_hold = hold_times[hold_times["key_code"] == key]["value"]
        if len(key_hold) > 0:
            features_dict[f"hold_{key}_mean"] = key_hold.mean()
            features_dict[f"hold_{key}_std"] = (
                key_hold.std() if len(key_hold) > 1 else 0
            )
            features_dict[f"hold_{key}_min"] = key_hold.min()
            features_dict[f"hold_{key}_max"] = key_hold.max()

    # FLIGHT
    for i in range(len(df) - 1):
        if df.iloc[i]["EVENT"] == "UP" and df.iloc[i + 1]["EVENT"] == "DOWN":
            first_key = df.iloc[i]["key_code"]
            second_key = df.iloc[i + 1]["key_code"]
            key_pair = f"{first_key}_{second_key}"

            if df.iloc[i + 1]["timing_type"] == "FLIGHT":
                flight_time = df.iloc[i + 1]["value"]

                feat_name = f"flight_{key_pair}"
                if feat_name in features_dict:
                    features_dict[feat_name] = (
                        features_dict[feat_name] + flight_time
                    ) / 2
                else:
                    features_dict[feat_name] = flight_time

    features_dict["user_id"] = user_id

    return pd.DataFrame([features_dict])


def main():
    data_dir = "keystroke_data"

    raw_data = load_keystroke_data(data_dir)

    all_features = []
    all_labels = []

    labels = []

    for user_id in raw_data["user_id"].unique():
        user_data = raw_data[raw_data["user_id"] == user_id]
        user_features = extract_keystroke_features(user_data, user_id)

        # 正例（本人）
        rows = len(user_features)
        all_features.append(user_features)
        all_labels.extend([1] * rows)

        # 負例（他人）- 簡易的に特徴をランダム変更
        for _ in range(3):  # 各ユーザーに対して3つの負例を生成
            noisy_features = user_features.copy()

            numeric_cols = noisy_features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noisy_features[col] *= np.random.uniform(0.7, 1.3)

            all_features.append(noisy_features)
            all_labels.extend([0] * rows)

    # 特徴とラベルの準備
    features_df = pd.concat(all_features).fillna(0)
    if "user_id" in features_df.columns:
        features_df = features_df.drop("user_id", axis=1)
    features = features_df.values
    labels = np.array(all_labels)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 正規化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデル構築
    models = {
        "lstm": build_lstm_model(input_shape=(X_train_scaled.shape[1], 1)),
        "cnn": build_cnn_model(input_shape=(X_train_scaled.shape[1], 1)),
        "hybrid": build_hybrid_model(input_shape=(X_train_scaled.shape[1], 1)),
    }

    # モデル学習
    for name, model in models.items():
        print(f"Training {name} model...")
        X_train_reshaped = X_train_scaled.reshape(
            X_train_scaled.shape[0], X_train_scaled.shape[1], 1
        )
        X_test_reshaped = X_test_scaled.reshape(
            X_test_scaled.shape[0], X_test_scaled.shape[1], 1
        )
        model.fit(
            X_train_reshaped,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
        )

    # モデル評価
    results = evaluate_models(models, X_test_reshaped, y_test)
    print("Model evaluation results:")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}, EER = {result['eer']:.4f}")

    # 最適モデルの選択
    best_model_name = min(results, key=lambda x: results[x]["eer"])
    best_model = models[best_model_name]
    best_threshold = results[best_model_name]["threshold"]
    print(f"Best model: {best_model_name} with threshold {best_threshold:.4f}")

    # TFLiteへの変換
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

    # TFLiteでLSTMを扱う設定
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True

    tflite_model = converter.convert()

    with open("keystroke_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved to keystroke_model.tflite")


if __name__ == "__main__":
    main()
