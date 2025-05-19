import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Установка пути к результатам
RESULTS_PATH = r".\results"

def visualize_results():
    try:
        # Проверка существования папки
        if not os.path.exists(RESULTS_PATH):
            print(f"❌ Папка не найдена: {RESULTS_PATH}")
            return

        # Поиск последнего JSON-файла
        json_files = [f for f in os.listdir(RESULTS_PATH) 
                    if f.startswith('results_') and f.endswith('.json')]
        
        if not json_files:
            print(f"❌ В папке нет JSON-файлов с результатами: {RESULTS_PATH}")
            return

        latest_json = max(json_files, key=lambda f: os.path.getmtime(os.path.join(RESULTS_PATH, f)))
        file_path = os.path.join(RESULTS_PATH, latest_json)

        # Загрузка данных
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверка структуры данных
        if 'articles' not in data:
            print("❌ Неверный формат файла: отсутствует ключ 'articles'")
            return

        df = pd.DataFrame(data['articles'])

        # Создание фигуры с 4 графиками
        plt.figure(figsize=(16, 12))
        plt.suptitle('Анализ результатов извлечения текста', fontsize=16, y=1.02)

        # График 1: Статус обработки (улучшенный)
        plt.subplot(2, 2, 1)
        status_counts = df['status'].value_counts()
        if len(status_counts) > 0:
            colors = ['#4CAF50' if s == 'success' else '#F44336' for s in status_counts.index]
            bars = plt.bar(status_counts.index, status_counts.values, color=colors)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height} ({height/len(df)*100:.1f}%)',
                        ha='center', va='bottom')
            
            plt.title('Статус обработки статей', pad=20)
            plt.ylabel('Количество статей')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'Нет данных о статусах', ha='center', va='center')
            plt.title('Статус обработки статей')

        # График 2: Распределение сходства (гистограмма с KDE)
        plt.subplot(2, 2, 2)
        if 'success' in status_counts and 'similarity' in df.columns:
            success_df = df[df['status'] == 'success']
            if len(success_df) > 0:
                # Гистограмма
                n, bins, patches = plt.hist(success_df['similarity'], bins=15, 
                                           color='#2196F3', edgecolor='white', 
                                           density=True, alpha=0.7)
                
                # KDE оценка
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(success_df['similarity'])
                x = np.linspace(success_df['similarity'].min(), success_df['similarity'].max(), 200)
                plt.plot(x, kde(x), color='#0D47A1', linewidth=2)
                
                plt.title('Распределение процента сходства', pad=20)
                plt.xlabel('Процент сходства')
                plt.ylabel('Плотность')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Добавляем среднее и медиану
                mean_val = success_df['similarity'].mean()
                median_val = success_df['similarity'].median()
                plt.axvline(mean_val, color='#FF5722', linestyle='--', linewidth=1.5)
                plt.axvline(median_val, color='#9C27B0', linestyle='--', linewidth=1.5)
                plt.legend(['KDE', f'Среднее: {mean_val:.1f}%', f'Медиана: {median_val:.1f}%'])
            else:
                plt.text(0.5, 0.5, 'Нет успешно обработанных статей', ha='center', va='center')
                plt.title('Распределение сходства')
        else:
            plt.text(0.5, 0.5, 'Нет данных о сходстве', ha='center', va='center')
            plt.title('Распределение сходства')

        # График 3: Сравнение длин текста (улучшенный)
        plt.subplot(2, 2, 3)
        if 'success' in status_counts and all(col in df.columns for col in ['original_length', 'lib_length']):
            success_df = df[df['status'] == 'success']
            if len(success_df) > 0:
                plt.scatter(success_df['original_length'], success_df['lib_length'], 
                           alpha=0.6, color='#009688')
                
                max_len = max(success_df['original_length'].max(), 
                             success_df['lib_length'].max())
                plt.plot([0, max_len], [0, max_len], 'r--', linewidth=1.5)
                
                # Вычисляем R²
                from sklearn.metrics import r2_score
                r2 = r2_score(success_df['original_length'], success_df['lib_length'])
                
                plt.title(f'Сравнение длины текста\n(R² = {r2:.2f})', pad=20)
                plt.xlabel('Длина оригинала (символов)')
                plt.ylabel('Длина извлечённого (символов)')
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Нет данных для сравнения', ha='center', va='center')
                plt.title('Сравнение длины текста')
        else:
            plt.text(0.5, 0.5, 'Нет данных о длинах текста', ha='center', va='center')
            plt.title('Сравнение длины текста')

        # График 4: Дополнительная аналитика
        plt.subplot(2, 2, 4)
        info_text = ""
        
        if 'metadata' in data:
            info_text += f"Дата анализа: {data['metadata'].get('generated_at', 'N/A')}\n"
            info_text += f"Источник данных: {data['metadata'].get('source_csv', 'N/A')}\n\n"
        
        info_text += f"Всего статей: {len(df)}\n"
        info_text += f"Успешно обработано: {status_counts.get('success', 0)}\n"
        info_text += f"Ошибок обработки: {status_counts.get('error', 0)}\n\n"
        
        if 'success' in status_counts and 'similarity' in df.columns:
            success_df = df[df['status'] == 'success']
            if len(success_df) > 0:
                info_text += "Статистика сходства:\n"
                info_text += f"• Среднее: {success_df['similarity'].mean():.1f}%\n"
                info_text += f"• Медиана: {success_df['similarity'].median():.1f}%\n"
                info_text += f"• Максимум: {success_df['similarity'].max():.1f}%\n"
                info_text += f"• Минимум: {success_df['similarity'].min():.1f}%\n"
                info_text += f"• Стандартное отклонение: {success_df['similarity'].std():.1f}%\n"
        
        plt.text(0.1, 0.9, info_text, ha='left', va='top', fontsize=12,
                bbox=dict(facecolor='#f5f5f5', edgecolor='#e0e0e0', boxstyle='round'))
        plt.axis('off')
        plt.title('Сводная информация', pad=20)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_PATH, 'visual_summary_enhanced.png')
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"✅ Улучшенная визуализация сохранена: {plot_path}")

    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {str(e)}")

if __name__ == "__main__":
    # Убираем предупреждение QT
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    print(f"🔍 Анализ результатов из папки: {RESULTS_PATH}")
    visualize_results()
    input("Нажмите Enter для выхода...")