import torch
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from onnxconverter_common import float16

yolo = YOLO("yolov8n.pt")
pt_model = yolo.model

dummy_input = torch.randn(1, 3, 640, 640)  # batch x каналы x высота x ширина

onnx_output_path = "model_fp32.onnx"
torch.onnx.export(
    pt_model,
    dummy_input,
    onnx_output_path,
    export_params=True,  # сохраняем веса
    opset_version=13,    # версия ONNX
    do_constant_folding=True,  # оптимизация констант
    input_names=["images"],    # имя входного слоя
    output_names=["output"],   # имя выходного слоя
    dynamic_axes=None          # фиксированный размер батча
)

print("Модель успешно экспортирована в ONNX")


onnx_model = onnx.load(onnx_output_path)
onnx.checker.check_model(onnx_model)


transposed_model = float16.convert_float_to_float16(onnx_model)
onnx.save(transposed_model, "model_fp16.onnx")
print("Модель успешно конвертирована в FP16")

# Предикт через PyTorch
with torch.no_grad(): # не обучаем, а тестируем, поэтому вырубаем запоминание метаданных
    pt_outputs = pt_model(dummy_input)  # Получаем tuple

if isinstance(pt_outputs, tuple):
    pt_output = pt_outputs[0].detach().cpu().numpy()  # берем первый выход
else:
    pt_output = pt_outputs.detach().cpu().numpy()

ort_session = ort.InferenceSession("model_fp32.onnx")
outputs = ort_session.run(None, {"images": dummy_input.numpy()})
ort_output_fp32 = outputs[0]


ort_session_fp16 = ort.InferenceSession("model_fp16.onnx")
outputs_fp16 = ort_session_fp16.run(
    None, {"images": dummy_input.numpy().astype(np.float16)}
)
ort_output_fp16 = outputs_fp16[0]


print("\nРезультаты:")
print("PyTorch output:", pt_output)
print("ONNX FP32 output:", ort_output_fp32)
print("ONNX FP16 output:", ort_output_fp16)
print("Разница FP32 vs PyTorch:", np.abs(pt_output - ort_output_fp32).max())
print("Разница FP16 vs PyTorch:", np.abs(pt_output.astype(np.float16) - ort_output_fp16).max())



import time

# Функция для замера скорости
def benchmark(session, input_data, runs=100):
    # Прогрев
    session.run(None, {"images": input_data})

    # Замер времени
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"images": input_data})
    end = time.perf_counter()

    return (end - start) / runs

# Функция для бенчмарка PyTorch
def benchmark_torch(model, input_data, runs=100):
    # Прогрев
    with torch.no_grad():
        model(input_data)

    # Замер времени
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            model(input_data)
    end = time.perf_counter()

    return (end - start) / runs

# Перемещаем модель и данные на CPU для честного сравнения
pt_model_cpu = pt_model.to('cpu')
dummy_input_cpu = dummy_input.to('cpu')

# Бенчмаркинг PyTorch
pt_time = benchmark_torch(pt_model_cpu, dummy_input_cpu)

# Для ONNX явно указываем использование CPU
ort_session = ort.InferenceSession("model_fp32.onnx", providers=['CPUExecutionProvider'])
ort_session_fp16 = ort.InferenceSession("model_fp16.onnx", providers=['CPUExecutionProvider'])

# Бенчмаркинг ONNX
fp32_time = benchmark(ort_session, dummy_input.numpy())
fp16_time = benchmark(ort_session_fp16, dummy_input.numpy().astype(np.float16))

print("\nСкорость выполнения:")
print(f"PyTorch CPU: {pt_time * 1000:.2f} мс/запрос")
print(f"ONNX FP32: {fp32_time * 1000:.2f} мс/запрос")
print(f"ONNX FP16: {fp16_time * 1000:.2f} мс/запрос")
print(f"Ускорение ONNX FP32 vs PyTorch: {pt_time / fp32_time:.2f}x")

from scipy.spatial.distance import cosine

def calculate_metrics(ref, test):
    abs_diff = np.abs(ref - test)
    return {
        "max_diff": np.max(abs_diff),
        "mean_diff": np.mean(abs_diff),
        "cosine_sim": 1 - cosine(ref.flatten(), test.flatten()),
        "mse": np.mean((ref - test) ** 2)
    }

metrics_fp32 = calculate_metrics(pt_output, ort_output_fp32)
metrics_fp16 = calculate_metrics(pt_output.astype(np.float16), ort_output_fp16)

print("\nСравнение точности:")
print("FP32 (ONNX vs PyTorch):")
for k, v in metrics_fp32.items():
    print(f"{k:>12}: {v:.6f}")

print("\nFP16 (ONNX vs PyTorch):")
for k, v in metrics_fp16.items():
    print(f"{k:>12}: {v:.6f}")
