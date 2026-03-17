import os

def print_project_structure(path='.', ignore_folders=None, ignore_files=None, max_depth=None):
    if ignore_folders is None:
        ignore_folders = {'__pycache__', '.git', 'venv', 'env', '.venv', 'node_modules', '.idea', '.vscode', 'raw'}
    if ignore_files is None:
        ignore_files = {'*.pyc', '*.pyo', '*.so', '*.dll'}
    
    def should_ignore(name, is_dir):
        if is_dir and name in ignore_folders:
            return True
        for pattern in ignore_files:
            if name.endswith(pattern.lstrip('*')):
                return True
        return False

    def print_tree(current_path, prefix='', depth=0):
        if max_depth is not None and depth >= max_depth:
            return
        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            return

        dirs = []
        files = []
        for entry in entries:
            full_path = os.path.join(current_path, entry)
            if should_ignore(entry, os.path.isdir(full_path)):
                continue
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                files.append(entry)

        for i, name in enumerate(dirs + files):
            is_last = (i == len(dirs + files) - 1)
            connector = '└── ' if is_last else '├── '
            print(prefix + connector + name)
            if name in dirs:
                extension = '    ' if is_last else '│   '
                print_tree(os.path.join(current_path, name), prefix + extension, depth + 1)

    print(os.path.basename(os.path.abspath(path)) + '/')
    print_tree(path)

# Пример использования
if __name__ == '__main__':
    print_project_structure('.')