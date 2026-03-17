import os
from pathlib import Path

def collect_project_files(output_filename='project_dump.txt', root_dir=None):
    # Если корневая директория не указана, используем текущую папку скрипта
    if root_dir is None:
        root_dir = Path(__file__).parent.resolve()
    else:
        root_dir = Path(root_dir).resolve()

    # Расширения файлов, которые нужно собрать
    target_extensions = {'.py', '.yaml', '.yml'}
    
    # Список файлов для игнорирования (чтобы не зациклиться и не мусорить)
    ignore_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', '.idea', '.vscode', 'processors'}
    ignore_files = {output_filename, 'collect_code.py'} # Игнорируем сам скрипт и выходной файл

    collected_content = []
    file_count = 0

    print(f"Начинаю сканирование папки: {root_dir}")

    # Проход по всем файлам рекурсивно
    for path in root_dir.rglob('*'):
        if path.is_file():
            # Проверка расширения
            if path.suffix.lower() not in target_extensions:
                continue
            
            # Проверка на игнорируемые файлы
            if path.name in ignore_files:
                continue

            # Проверка на игнорируемые папки (если часть пути содержит имя игнорируемой папки)
            if any(ignore_part in path.parts for ignore_part in ignore_dirs):
                continue

            try:
                # Чтение содержимого файла
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Вычисляем относительный путь от корня проекта для красоты вывода
                relative_path = path.relative_to(root_dir)
                
                # Формируем блок текста
                header = f"\n{'='*60}\nFILE: {relative_path}\n{'='*60}\n"
                block = f"{header}{content}\n"
                
                collected_content.append(block)
                file_count += 1
                print(f"Добавлен: {relative_path}")
                
            except UnicodeDecodeError:
                print(f"Пропущен (кодировка): {path}")
            except Exception as e:
                print(f"Ошибка при чтении {path}: {e}")

    # Запись всего в один файл
    output_path = root_dir / output_filename
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PROJECT DUMP\nRoot: {root_dir}\nTotal files: {file_count}\n")
            f.write("="*60 + "\n\n")
            f.writelines(collected_content)
        
        print(f"\nГотово! Весь код сохранен в файл: {output_path}")
        print(f"Всего файлов обработано: {file_count}")
    except Exception as e:
        print(f"Ошибка при записи итогового файла: {e}")

if __name__ == "__main__":
    # Можно передать путь к проекту как аргумент, если запускаете из другого места
    # Например: python collect_code.py /path/to/my/project
    import sys
    if len(sys.argv) > 1:
        collect_project_files(root_dir=sys.argv[1])
    else:
        collect_project_files()