import os
from pathlib import Path
from moviepy import VideoFileClip

# Ввод папки
folder = input("Введите путь к папке: ").strip()

# Проверяем существование папки
if not os.path.exists(folder):
    print("Папка не найдена!")
    exit()

# Создаем папку для GIF
gif_folder = os.path.join(folder, "converted_gifs")
os.makedirs(gif_folder, exist_ok=True)

# Ищем все MP4 файлы
mp4_files = list(Path(folder).glob("*.mp4"))

if not mp4_files:
    print("MP4 файлы не найдены!")
    exit()

print(f"Найдено {len(mp4_files)} MP4 файлов")

# Конвертируем каждый файл
for mp4 in mp4_files:
    output_gif = os.path.join(gif_folder, mp4.stem + ".gif")
    print(f"Конвертирую: {mp4.name}")

    try:
        # Загружаем видео
        video = VideoFileClip(str(mp4))

        # Сохраняем как GIF (без параметра program)
        video.write_gif(output_gif, fps=video.fps)

        # Закрываем видео
        video.close()
        print(f"  [OK] Готово: {mp4.name}")

    except Exception as e:
        print(f"  [ERROR] Ошибка с {mp4.name}: {e}")

print(f"\n[OK] Готово! Все GIF в папке: {gif_folder}")
input("\nНажмите Enter для выхода...")