# QIG_FlowshopScheduling

Repo triển khai các biến thể Iterated Greedy cho bài toán **Permutation Flow Shop Scheduling Problem (PFSP)** và bộ chạy benchmark trên các bộ dữ liệu **Taillard** và **VRF (VFR*_Gap)**.

## 1) Yêu cầu hệ thống

- Python: khuyến nghị **Python 3.10+** (3.8+ thường vẫn chạy được, nhưng nên dùng 3.10/3.11 để ổn định).
- Hệ điều hành: Windows / macOS / Linux đều được.
Nếu bạn gặp lỗi import khi chạy, hãy xem mục **Troubleshooting** ở cuối README.

## 2) Clone repo

```bash
git clone https://github.com/PhanVuQuocThang/QIG_FlowshopScheduling.git
cd QIG_FlowshopScheduling
```

## 3) Tạo môi trường ảo và cài dependencies

### 3.1) macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy
```

### 3.2) Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy
```

## 4) Cấu trúc thư mục dữ liệu (datasets)

Các script trong repo mặc định đọc dữ liệu từ thư mục `datasets/` với cấu trúc sau (đúng theo `benchmark.py`):

```
datasets/
├── bounds/
│   ├── Taillard_UB_Schedules_v9.csv
│   ├── VFRlarge_UB_Schedules_v9.csv
│   └── VFRsmall_UB_Schedules_v9.csv
├── taillard_instances/
│   ├── ta001
│   ├── ta002
│   └── ...
└── vrf_instances/
    ├── Large/
    │   ├── VFR100_20_1_Gap.txt
    │   └── ...
    └── Small/
        ├── VFR10_5_1_Gap.txt
        └── ...
```

Repo đã có sẵn các thư mục và file bounds trong `datasets/bounds/`.
Nếu bạn thay đổi đường dẫn dữ liệu, hãy dùng tham số `--root` (xem phần chạy benchmark).

## 5) Chạy nhanh: thử 1 instance (khuyến nghị để kiểm tra setup)

Script: `try_single_instance.py`

Mặc định script sẽ:
- load dataset từ `datasets/`
- chọn instance Taillard có tên `ta036`
- chạy các chiến lược:
  - `QIG` (qlearning)
  - `IIG1`, `IIG2`, `IIG3` (individual với d=1..3)
  - `RIG` (random)

Chạy:

```bash
python try_single_instance.py
```

Kết quả sẽ in ra:
- thông tin instance (n, m)
- time limit (ms) tính theo công thức trong `benchmark.py`
- Cmax và RPD (nếu instance có UB)

Lưu ý:
- `try_single_instance.py` đang chọn cứng instance `ta036`. Nếu bạn không có instance này trong `datasets/taillard_instances/` thì script sẽ lỗi.
- Bạn có thể sửa dòng chọn instance trong file hoặc đổi sang chạy benchmark có tham số `--limit`.

## 6) Chạy benchmark và xuất kết quả CSV (thực nghiệm hàng loạt)

Script: `experiment_runner.py`

### 6.1) Quick test (chạy ít lần, ít instance)

Ví dụ chạy 3 lần/instance, time-scale t=60, xuất kết quả vào `results/`:

```bash
python experiment_runner.py --root datasets --runs 3 --t 60 --out results
```

Nếu muốn chạy ít instance để test nhanh (debug), dùng `--limit`:

```bash
python experiment_runner.py --root datasets --runs 1 --t 60 --out results --limit 2
```

### 6.2) Chọn dataset

`experiment_runner.py` hỗ trợ `--dataset`:

- `taillard`
- `vrf_small`
- `vrf_large`
- `all` (mặc định)

Ví dụ chỉ chạy Taillard:

```bash
python experiment_runner.py --root datasets --dataset taillard --runs 3 --t 60 --out results
```

### 6.3) Chọn thuật toán chạy

Mặc định `--algos all` sẽ chạy:
- `IIG1`, `IIG2`, `IIG3`, `IIG4`
- `RIG`
- `QIG`

Bạn có thể chọn subset:

```bash
python experiment_runner.py --root datasets --dataset taillard --runs 5 --t 60 --out results --algos QIG,RIG
```

### 6.4) Ý nghĩa tham số thời gian `--t`

Trong `benchmark.py`, giới hạn thời gian (ms) được tính:

- `time_limit_ms(n, m, t) = int(n * m / 2 * t)`

Trong đó `t` thường lấy {60, 90, 120}.

Ví dụ chạy theo “full paper” (nếu bạn muốn tái lập đúng cấu hình này):

```bash
python experiment_runner.py --root datasets --runs 30 --t 60  --out results
python experiment_runner.py --root datasets --runs 30 --t 90  --out results
python experiment_runner.py --root datasets --runs 30 --t 120 --out results
```

### 6.5) Output

`experiment_runner.py` sẽ tạo file CSV trong thư mục `--out`, tên dạng:

- `raw_results_t{t}_{YYYYMMDD_HHMMSS}.csv`

Mỗi dòng là một lần chạy: instance × algorithm × run, có các cột:
- `instance, n, m, algorithm, run_id, seed, cmax, upper_bound, rpd, cpu_sec, time_scale`

### 6.6) Chạy im lặng (ít log)

```bash
python experiment_runner.py --root datasets --runs 3 --t 60 --out results --quiet
```

## 7) Ghi chú về format dữ liệu instance (nếu bạn muốn tự thêm instance)

`benchmark.py` hỗ trợ các raw format chính:

### 7.1) Taillard

Có 2 dạng:

- Format A (machine-major): header có `n m seed UB LB`, sau đó `m` dòng, mỗi dòng có `n` thời gian xử lý.
- Format B (job-pair kiểu `ta001/ta002`): header `n m`, sau đó `n` dòng (mỗi job), mỗi dòng có `m` cặp `(machine_id, processing_time)`.

Dữ liệu được chuyển về ma trận nội bộ `p[machine][job]`.

### 7.2) VRF/VFR *_Gap.txt

- Header: `n m`
- Sau đó `n` dòng, mỗi dòng chứa `m` cặp `(machine_id, processing_time)`.

File name có thể là `VFR..._Gap.txt` (loader có xử lý bỏ hậu tố `_Gap` khi đặt tên instance).

## 8) Troubleshooting

### 8.1) `try_single_instance.py` lỗi vì không tìm thấy instance `ta036`

Nguyên nhân: trong `datasets/taillard_instances/` không có file tương ứng để loader tạo instance tên `ta036`.

Cách xử lý:
- Cung cấp đúng bộ dữ liệu Taillard có chứa `ta036`, hoặc
- Sửa script `try_single_instance.py` để chọn instance khác đang có thật trong thư mục, hoặc
- Dùng `experiment_runner.py` kèm `--limit` để test nhanh mà không phụ thuộc vào tên `ta036`.

### 8.2) Sai cấu trúc thư mục datasets

Nếu bạn đặt dữ liệu ở chỗ khác, hãy trỏ lại bằng:

```bash
python experiment_runner.py --root <duong_dan_dataset_cua_ban> ...
```

Hoặc trong code, chỗ load dataset dùng:
- `load_project_datasets("datasets")`

## 9) Các file chính trong repo (tóm tắt)

- `try_single_instance.py`: chạy thử 1 instance để xem kết quả nhanh.
- `experiment_runner.py`: chạy benchmark hàng loạt, xuất CSV.
- `benchmark.py`: loader dữ liệu + tính makespan + time limit + utilities benchmark.
- `ig.py`, `strategy.py`, `operators.py`, `acceptance.py`, `initialization.py`, `solution.py`: các thành phần của solver Iterated Greedy / QIG.
- `visualize.py`, `report.py`, `metrics.py`: các tiện ích phụ trợ (nếu bạn muốn mở rộng pipeline phân tích/visualize).
