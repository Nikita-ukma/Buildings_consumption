import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Завантаження та підготовка даних
def load_data(file_path):
    """Завантаження даних з CSV файлу"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # Якщо надано дані у текстовому форматі, конвертуємо їх в датафрейм
        lines = file_path.strip().split('\n')
        header = lines[0].split(',')
        data = []
        for line in lines[1:]:
            data.append(line.split(','))
        df = pd.DataFrame(data, columns=header)
    
    # Конвертуємо текстові стовпці в числові, де можливо
    for col in df.columns:
        if col != 'Timestamp' and col != 'Building Type' and col != 'Carbon Emission Reduction Category' and col != 'Occupancy Schedule' and col != 'Building Orientation' and col != 'Maintenance Status' and col != 'Demand Response Participation':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Конвертуємо часову мітку в datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Обробка категоріальних змінних
    df = pd.get_dummies(df, columns=['Building Type', 'Building Orientation', 'Carbon Emission Reduction Category', 'Maintenance Status'], drop_first=True)
    
    # Бінарні змінні
    df['Power Outage Indicator'] = df['Power Outage Indicator'].astype(int)
    df['Demand Response Participation'] = df['Demand Response Participation'].astype(int)
    
    # Обробка Occupancy Schedule
    df['Is_Occupied'] = df['Occupancy Schedule'].apply(lambda x: 1 if x == 'Occupied' else 0)
    
    return df

def preprocess_data(df):
    """Підготовка даних для моделювання"""
    # Видалення стовпців, які не потрібні для моделювання
    cols_to_drop = ['Timestamp', 'Occupancy Schedule', 'Historical Energy Consumption (kWh)']
    df = df.drop(cols_to_drop, axis=1)
    
    # Обробка пропущених значень
    df = df.dropna()
    
    # Розділення на ознаки та цільову змінну
    X = df.drop('Energy Consumption (kWh)', axis=1)
    y = df['Energy Consumption (kWh)']
    
    return X, y

def train_and_evaluate_models(X, y):
    """Навчання та оцінка моделей"""
    # Розділення на навчальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабування даних
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ініціалізація моделей
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Навчання та оцінка кожної моделі
    for name, model in models.items():
        # Навчання моделі
        model.fit(X_train_scaled, y_train)
        
        # Прогнозування
        y_pred = model.predict(X_test_scaled)
        
        # Обчислення метрик
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
        
        # Візуалізація прогнозів vs реальні значення
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Реальне значення')
        plt.ylabel('Прогнозоване значення')
        plt.title(f'Реальні vs Прогнозовані значення - {name}')
        plt.savefig(f'predictions_{name.replace(" ", "_")}.png')
        plt.close()
    
    return results, models

def correlation_analysis(df):
    """Аналіз кореляцій між змінними"""
    # Вибір числових колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Обчислення кореляцій
    corr_matrix = df[numeric_cols].corr()
    
    # Візуалізація кореляційної матриці
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Кореляційна матриця')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Кореляції з цільовою змінною
    target_corr = corr_matrix['Energy Consumption (kWh)'].sort_values(ascending=False)
    
    # Візуалізація кореляцій з цільовою змінною
    plt.figure(figsize=(12, 8))
    target_corr.drop('Energy Consumption (kWh)').plot(kind='bar')
    plt.title('Кореляція з енергоспоживанням')
    plt.xlabel('Параметри')
    plt.ylabel('Коефіцієнт кореляції')
    plt.tight_layout()
    plt.savefig('target_correlation.png')
    plt.close()
    
    return target_corr

def feature_importance_analysis(models, X):
    """Аналіз важливості ознак"""
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Отримання важливості ознак
            importances = model.feature_importances_
            
            # Створення DataFrame з важливістю ознак
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Візуалізація важливості ознак
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
            plt.title(f'Топ-15 важливих ознак - {name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png')
            plt.close()
            
            print(f"\nВажливість ознак для {name}:")
            print(feature_importance_df.head(10))

def main():
    """Головна функція для виконання аналізу"""
    # Шлях до файлу даних (замініть на свій)
    data_file = 'data/electricity_dataset.csv'
    
    # Завантаження даних
    try:
        df = load_data(data_file)
        print(f"Дані успішно завантажено. Розмір датасету: {df.shape}")
    except Exception as e:
        print(f"Помилка завантаження даних: {e}")
        # Якщо файл не доступний, використовуємо дані з текстового виведення
        print("Використання даних з прикладу...")
        # Тут можна додати демо-дані, якщо потрібно
        return
    
    # Загальний опис датасету
    print("\nЗагальна інформація про датасет:")
    print(df.info())
    print("\nСтатистичний опис:")
    print(df.describe())
    
    # Аналіз кореляцій
    print("\nАналіз кореляцій:")
    target_corr = correlation_analysis(df)
    print("\nТоп-10 кореляцій з енергоспоживанням:")
    print(target_corr.head(10))
    
    # Підготовка даних для моделювання
    X, y = preprocess_data(df)
    
    # Навчання та оцінка моделей
    print("\nНавчання та оцінка моделей:")
    results, models = train_and_evaluate_models(X, y)
    
    # Виведення результатів
    for name, metrics in results.items():
        print(f"\nРезультати для {name}:")
        print(f"MAE: {metrics['MAE']:.2f} кВт⋅год")
        print(f"RMSE: {metrics['RMSE']:.2f} кВт⋅год")
        print(f"R²: {metrics['R²']:.4f}")
    
    # Аналіз важливості ознак
    print("\nАналіз важливості ознак:")
    feature_importance_analysis(models, X)
    
    # Висновок
    print("\nВисновок:")
    best_r2 = max([metrics['R²'] for metrics in results.values()])
    if best_r2 < 0.5:
        print("Результати моделювання підтверджують неможливість ефективного прогнозування")
        print("енергоспоживання будівель без врахування часових рядів.")
        print(f"Найкращий досягнутий R² складає лише {best_r2:.4f}, що є недостатнім")
        print("для практичного застосування в системах енергоменеджменту.")

if __name__ == "__main__":
    main()