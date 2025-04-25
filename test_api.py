import requests
import json
from datetime import datetime, timedelta
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_prediction(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
# URL của API (đổi nếu cần)
API_URL = "http://localhost:8000/predict"
true_cpu = []
true_ram = []
def generate_test_data(num_records=6):
    """Tạo dữ liệu test ngẫu nhiên nhưng hợp lý"""
    records = []
    base_time = datetime.now()
    
    for i in range(num_records):
        scantime = (base_time - timedelta(minutes=5*i)).isoformat()
        
        record = {
            "scantime": scantime,
            "cpu": {
                "utilization": random.uniform(10, 90),  # % CPU usage
                "processes": random.randint(50, 300)   # Số process
            },
            "ram": {
                "percent": random.uniform(20, 80)       # % RAM usage
            }
        }
        records.append(record)
        true_cpu.append(record["cpu"]["utilization"])
        true_ram.append(record["ram"]["percent"])
    
    return {"records": records}

def test_predict_api():
    """Kiểm thử API predict"""
    print("=== Bắt đầu kiểm thử API predict ===")
    
    # 1. Tạo dữ liệu test
    test_data = generate_test_data()
    print("\nDữ liệu test gửi đến API:")
    print(json.dumps(test_data, indent=2))
    
    # 2. Gọi API
    try:
        response = requests.post(API_URL, json=test_data)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        
        # 3. Kiểm tra kết quả
        result = response.json()
        print("\nKết quả nhận được từ API:")
        print(json.dumps(result, indent=2))
        
        # Kiểm tra cấu trúc response
        assert "predicted_cpu" in result, "Thiếu predicted_cpu trong response"
        assert "predicted_ram" in result, "Thiếu predicted_ram trong response"
        assert len(result["predicted_cpu"]) == 6, "Số lượng dự đoán CPU không đúng"
        assert len(result["predicted_ram"]) == 6, "Số lượng dự đoán RAM không đúng"
        
        print("\n✅ Kiểm thử thành công! Response hợp lệ.")
        cpu_metrics = evaluate_prediction(true_cpu, result["predicted_cpu"])
        ram_metrics = evaluate_prediction(true_ram, result["predicted_ram"])
        print("\n🎯 Đánh giá độ chính xác:")
        print("CPU:", cpu_metrics)
        print("RAM:", ram_metrics)
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Lỗi khi gọi API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Chi tiết lỗi: {e.response.text}")
    except AssertionError as e:
        print(f"\n❌ Kiểm thử thất bại: {e}")

if __name__ == "__main__":
    test_predict_api()