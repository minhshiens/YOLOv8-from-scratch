# YOLOv8-from-scratch (FCOS Object Detector)

Mô hình Object Detection được xây dựng từ đầu (from scratch) dựa trên kiến trúc FCOS (Fully Convolutional One-Stage Object Detection) với backbone là ResNet-50. Mô hình dự đoán 5 lớp đối tượng: `person`, `car`, `dog`, `cat`, `chair`.

## 1. Cách cài đặt môi trường

Cài đặt các thư viện cần thiết thông qua file `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Cách huấn luyện (Training)

Chạy lệnh sau để bắt đầu quá trình huấn luyện mô hình. Mã nguồn sẽ tự động lưu lại các epoch và giữ phiên bản tốt nhất dựa trên mAP của tập validation.

```bash
python train.py --train_data ./public/annotations/train.json --val_data ./public/annotations/val.json --image_dir ./public/train/images --val_image_dir ./public/val/images --checkpoint_dir ./models/
```

Mô hình tốt nhất sẽ được lưu tự động vào file `./models/best.pth`.

## 3. Cách chạy suy luận (Inference)

Để dự đoán trên một tập ảnh mới, sử dụng lệnh sau. Kết quả sẽ được xuất ra file `predictions.json` theo đúng định dạng yêu cầu:

```bash
python predict.py --image_dir /path/to/images --output predictions.json
```

*Lưu ý: Lệnh suy luận mặc định sẽ load trọng số từ `./models/best.pth` nếu không truyền thêm cờ `--checkpoint`.*

## 4. Vị trí đặt mô hình hoặc trọng số mô hình

* Trọng số tốt nhất của mô hình sau khi huấn luyện được lưu tại: `./models/best.pth`
* Trọng số ở epoch cuối cùng (latest) được lưu tại: `./models/latest.pth`
* File kết quả dự đoán và điểm đánh giá trên tập validation được lưu tại: `./models/results/val_predictions.json` và `./models/results/val_score.json`