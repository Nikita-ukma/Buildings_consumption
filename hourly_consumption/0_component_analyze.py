import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates


df = pd.read_csv('data/DAYTON_hourly.csv')


df['Datetime'] = pd.to_datetime(df['Datetime'])


duplicates = df['Datetime'].duplicated().sum()
print(f"Знайдено дублікатів дати: {duplicates}")

if duplicates > 0:
    print("Видаляємо дублікати...")
    df = df.drop_duplicates(subset=['Datetime'])


df.set_index('Datetime', inplace=True)


print(f"Пропущені значення: {df['DAYTON_MW'].isna().sum()}")
if df['DAYTON_MW'].isna().sum() > 0:
    df['DAYTON_MW'] = df['DAYTON_MW'].interpolate(method='time')


print("Базова статистика:")
print(df['DAYTON_MW'].describe())


df = df.sort_index()


plt.figure(figsize=(15, 6))
plt.plot(df['DAYTON_MW'], color='blue', label='Енергоспоживання (MW)')
plt.title('Часовий ряд енергоспоживання')
plt.xlabel('Дата')
plt.ylabel('Енергоспоживання (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('time_series.png')
plt.close()



time_diffs = df.index.to_series().diff().value_counts()
print("Найчастіші інтервали між вимірюваннями:")
print(time_diffs.head())



try:
    df_resampled = df.asfreq('H')
    print(f"Пропущені значення після ресемплінгу: {df_resampled['DAYTON_MW'].isna().sum()}")
    
    
    if df_resampled['DAYTON_MW'].isna().sum() > 0:
        df_resampled['DAYTON_MW'] = df_resampled['DAYTON_MW'].interpolate(method='time')
    
    
    df = df_resampled
except Exception as e:
    print(f"Помилка при ресемплінгу: {e}")
    print("Продовжуємо з оригінальними даними")



try:
    daily_decomposition = seasonal_decompose(df['DAYTON_MW'], model='additive', period=24)
    has_daily = True
except Exception as e:
    print(f"Помилка при добовій декомпозиції: {e}")
    has_daily = False


try:
    weekly_decomposition = seasonal_decompose(df['DAYTON_MW'], model='additive', period=168)
    has_weekly = True
except Exception as e:
    print(f"Помилка при тижневій декомпозиції: {e}")
    has_weekly = False


if len(df) >= 8760:  
    try:
        yearly_decomposition = seasonal_decompose(df['DAYTON_MW'], model='additive', period=8760)
        has_yearly = True
    except Exception as e:
        print(f"Помилка при річній декомпозиції: {e}")
        has_yearly = False
else:
    print("Недостатньо даних для річної декомпозиції")
    has_yearly = False


def plot_decomposition(decomposition, title_prefix):
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    
    decomposition.observed.plot(ax=axes[0], color='blue')
    axes[0].set_title(f'{title_prefix} - Оригінальний ряд')
    axes[0].set_ylabel('MW')
    axes[0].grid(True)
    
    
    decomposition.trend.plot(ax=axes[1], color='green')
    axes[1].set_title(f'{title_prefix} - Тренд')
    axes[1].set_ylabel('MW')
    axes[1].grid(True)
    
    
    decomposition.seasonal.plot(ax=axes[2], color='red')
    axes[2].set_title(f'{title_prefix} - Сезонність')
    axes[2].set_ylabel('MW')
    axes[2].grid(True)
    
    
    decomposition.resid.plot(ax=axes[3], color='purple')
    axes[3].set_title(f'{title_prefix} - Залишки (шум)')
    axes[3].set_ylabel('MW')
    axes[3].grid(True)
    
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix.lower().replace(" ", "_")}_decomposition.png')
    plt.close()


if has_daily:
    plot_decomposition(daily_decomposition, "Добова декомпозиція")

if has_weekly:
    plot_decomposition(weekly_decomposition, "Тижнева декомпозиція")

if has_yearly:
    plot_decomposition(yearly_decomposition, "Річна декомпозиція")



if has_daily:
    def extract_cyclical_component(data, window=30*24):  
        
        
        trend = data.rolling(window=window, center=True).mean()
        
        detrended = data - trend
        
        seasonal = daily_decomposition.seasonal
        
        
        seasonal_aligned = seasonal.reindex(detrended.index, method='nearest')
        cyclical = detrended - seasonal_aligned
        return cyclical.dropna(), trend.dropna()

    
    window_sizes = [24*7, 24*30, 24*90]  
    plt.figure(figsize=(15, 12))

    for i, window in enumerate(window_sizes):
        try:
            cyclical, trend = extract_cyclical_component(df['DAYTON_MW'], window=window)
            
            plt.subplot(len(window_sizes), 1, i+1)
            plt.plot(cyclical, label=f'Циклічна компонента (вікно={window//24} днів)')
            plt.title(f'Циклічна компонента з вікном згладжування {window//24} днів')
            plt.ylabel('MW')
            plt.grid(True)
            plt.legend()
        except Exception as e:
            print(f"Помилка при аналізі циклічності з вікном {window}: {e}")

    plt.tight_layout()
    plt.savefig('cyclical_components.png')
    plt.close()


plt.figure(figsize=(15, 6))


plt.subplot(1, 2, 1)
plt.hist(df['DAYTON_MW'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Розподіл значень енергоспоживання')
plt.xlabel('Енергоспоживання (MW)')
plt.ylabel('Частота')
plt.grid(True)


try:
    from pandas.plotting import autocorrelation_plot
    plt.subplot(1, 2, 2)
    autocorrelation_plot(df['DAYTON_MW'].dropna())
    plt.title('Автокореляція енергоспоживання')
    plt.grid(True)
except Exception as e:
    print(f"Помилка при побудові автокореляції: {e}")

plt.tight_layout()
plt.savefig('distribution_autocorrelation.png')
plt.close()

print("Аналіз часового ряду завершено. Результати збережено в графічних файлах.")


result_df = pd.DataFrame({'Original': df['DAYTON_MW']})

if has_daily:
    result_df['Trend_Daily'] = daily_decomposition.trend
    result_df['Seasonal_Daily'] = daily_decomposition.seasonal
    result_df['Residual_Daily'] = daily_decomposition.resid

if has_weekly:
    result_df['Trend_Weekly'] = weekly_decomposition.trend
    result_df['Seasonal_Weekly'] = weekly_decomposition.seasonal
    result_df['Residual_Weekly'] = weekly_decomposition.resid

if has_yearly:
    result_df['Trend_Yearly'] = yearly_decomposition.trend
    result_df['Seasonal_Yearly'] = yearly_decomposition.seasonal
    result_df['Residual_Yearly'] = yearly_decomposition.resid

result_df.to_csv('timeseries_components.csv')
print("Компоненти часового ряду збережено у файл 'timeseries_components.csv'")


with open('report_data.txt', 'w') as f:
    f.write("АНАЛІЗ ЧАСОВОГО РЯДУ ЕНЕРГОСПОЖИВАННЯ\n")
    f.write("======================================\n\n")
    f.write(f"Кількість записів: {len(df)}\n")
    f.write(f"Період даних: з {df.index.min()} по {df.index.max()}\n\n")
    f.write("Базова статистика:\n")
    f.write(str(df['DAYTON_MW'].describe()) + "\n\n")
    
    if has_daily:
        f.write("Добова сезонність:\n")
        f.write(f"Мін. значення: {daily_decomposition.seasonal.min()}\n")
        f.write(f"Макс. значення: {daily_decomposition.seasonal.max()}\n")
        f.write(f"Розмах: {daily_decomposition.seasonal.max() - daily_decomposition.seasonal.min()}\n\n")
    
    if has_weekly:
        f.write("Тижнева сезонність:\n")
        f.write(f"Мін. значення: {weekly_decomposition.seasonal.min()}\n")
        f.write(f"Макс. значення: {weekly_decomposition.seasonal.max()}\n")
        f.write(f"Розмах: {weekly_decomposition.seasonal.max() - weekly_decomposition.seasonal.min()}\n\n")

print("Звіт було збережено у файл 'report_data.txt'")