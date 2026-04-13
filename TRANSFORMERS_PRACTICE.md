### Практика про трансформерам

#### Скачайте проект

```bash
git clone https://github.com/runnerup96/minGPT
cd minGPT
```

#### Установите окружение

```bash
# activate/install Python 3.11

# Создайте окружение
python -m venv .venv 

# Активируйте его
source 

# Установите библиотеки -- для скорости установки PyTorch версия CPU
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Запустите проверку импортов -- если ошибки не будет, все хорошо.
python projects/overfit_test/test_environment.py 
```

#### Практика

```bash
# Реализуйте все Code Here строки кода

# Запустите 
python projects/overfit_test/run_overfit.py

# Ошибка должна упасть до 10^-5, а модель должна корректно воспроизвести обучающую выборку в таком формате.
```

Такой результат:


| Source          | Expected        | Predicted       | OK?  |
| --------------- | --------------- | --------------- | ---- |
| [3, 1, 4, 1, 5] | [5, 1, 4, 1, 3] | [5, 1, 4, 1, 3] | PASS |
| [2, 7, 1, 8, 2] | [2, 8, 1, 7, 2] | [2, 8, 1, 7, 2] | PASS |
| [6, 6, 2, 8, 3] | [3, 8, 2, 6, 6] | [3, 8, 2, 6, 6] | PASS |
| [1, 4, 1, 4, 2] | [2, 4, 1, 4, 1] | [2, 4, 1, 4, 1] | PASS |


