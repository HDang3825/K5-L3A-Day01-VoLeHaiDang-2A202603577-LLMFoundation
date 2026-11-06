# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603577
- **Họ và tên:** Võ Lê Hải Đăng
- **SEED:** 7342

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán lọc SMS Spam/Ham từ kết quả chạy thực nghiệm của SEED 7342:

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.821 | 0.902 | 0.889 | 5.33 | 15,418.7 | 0.06 | 0.23 | $0.000 |
| **2. BoW + MLP (deep learning)** | 1.000 | 0.937 | 0.967 | 0.947 | 7.59 | 25,843.8 | 0.28 | 0.60 | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.845 | 0.979 | 0.907 | 0.950 | 110.97 | 19.8 | 50.33 | 67.93 | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.949 | 0.979 | 0.964 | 0.950 | 111.38 | 31.2 | 28.81 | 40.33 | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.950 | 1.000 | 0.974 | 0.976 | 111.93 | 31.1 | 28.30 | 39.18 | $0.000 |
| **4a. LLM nhỏ zero-shot** | 1.000 | 1.000 | 1.000 | 1.000 | 0.00 | 1.20 | 647.34 | 2,119.86 | $0.020 |
| **4c. LLM nhỏ few-shot** | 0.952 | 1.000 | 0.976 | 0.976 | 0.00 | 0.83 | 598.39 | 2,088.94 | $0.048 |

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất so với kết quả:

1. **`hard_eval` (Dự đoán accuracy trên 50 case khó):**
   - *Dự đoán:* Accuracy giảm khoảng 15% đến 30% (tức còn 70% – 85%).
   - *Thực tế:* Mô hình BoW tụt dốc không phanh: F1 giảm từ 0.889 xuống còn **0.308** và accuracy chỉ đạt **50.9%** (tụt gần 50%).
   - *Nguyên nhân:* Bộ 50 case khó chứa các bẫy tinh vi như né từ khóa (obfuscation) và spam giọng tự nhiên (natural spam) hoàn toàn nằm ngoài từ điển của BoW, khiến BoW gần như mất khả năng nhận diện.

2. **`llm_fewshot` (So sánh few-shot với zero-shot của LLM):**
   - *Dự đoán:* Few-shot sẽ giúp F1 tăng lên so với zero-shot.
   - *Thực tế:* Zero-shot đạt F1 tuyệt đối 1.000 trên tập SUB 40 tin, trong khi Few-shot lại đạt **0.976** (bị giảm nhẹ do 1 tin nhắn bị phân loại nhầm) và chi phí tăng hơn gấp đôi ($0.048 so với $0.020 / 1k tin).
   - *Nguyên nhân:* Trên tập SUB nhỏ (40 tin), zero-shot của model hiện đại đã rất thông minh. Khi thêm ví dụ few-shot vào prompt, context dài hơn làm tăng chi phí và đôi khi gây ra hiện tượng thiên lệch (bias) ngoài ý muốn cho một số trường hợp biên.

3. **`dl_curve` (MLP so với Logistic Regression):**
   - *Dự đoán:* MLP thêm lớp ẩn phi tuyến sẽ vượt trội rõ rệt so với Logistic Regression.
   - *Thực tế:* Trên không gian Embedding (3b vs 3c), MLP head (F1 = 0.974) chỉ nhỉnh hơn LogReg (F1 = 0.964) đúng 0.010 điểm F1; trong khi đó chi phí train và tính toán phức tạp hơn.
   - *Nguyên nhân:* Không gian vector embedding của MiniLM đã được tiền huấn luyện rất tối ưu và gần như phân tách tuyến tính, do đó bộ phân loại tuyến tính đơn giản như LogReg đã khai thác gần như trọn vẹn đặc trưng, lớp ẩn phi tuyến không tạo ra khác biệt lớn.

---

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** Mô hình **1. BoW + LogReg** (F1 tụt từ 0.889 xuống 0.308) và **2. BoW + MLP** (F1 tụt từ 0.947 xuống 0.341).
- **Nhóm case gây tụt nặng nhất:** Nhóm `obfuscation` (né từ khoá) và `natural_spam` (spam giọng tự nhiên).

### Hai case cụ thể & cơ chế khiến model sai:
1. **Case #13 [Spam giọng tự nhiên] (Tất cả model truyền thống đều sai):**
   - *Nội dung:* `"Hope you're well! Quick reminder that your photos from the event are ready, view them at photos-cloud.com"` (Nhãn thật: `spam`).
   - *Cơ chế sai:* Tin nhắn được viết với giọng điệu tự nhiên, lịch sự như người quen gửi ảnh kỷ niệm, hoàn toàn không chứa từ khoá quảng cáo hay giục giã ("free", "win", "urgent"). Cả BoW lẫn Encoder đều không thấy đặc trưng lừa đảo truyền thống nên đều đoán nhầm thành `ham`. Chỉ có LLM với khả năng đọc hiểu ngữ cảnh sâu và nhận diện link khả nghi mới bắt được.
2. **Case #1 [Tưởng spam nhưng không phải]:**
   - *Nội dung:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: `ham`).
   - *Cơ chế sai:* Tin nhắn cá cược bạn bè thông thường nhưng chứa hàng loạt từ khoá "nhạy cảm": `Congrats`, `WON`, `Call`. Mô hình BoW cộng dồn trọng số dương của các từ này và kết luận ngay là `spam` (chặn nhầm), trong khi Encoder và LLM hiểu được từ "bet lol" và "owe you a beer" để nhận ra đây là tin nhắn cá nhân.

---

## 5. Thử thách cá nhân

- **Đề bài nhận được (Đề #5 theo SEED 7342):**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

- **4 tin nhắn tự thiết kế:**
  1. `Special gift reserved for your number, confirm your details at bit.ly/claim-item` (Spam né từ khoá)
  2. `Your package delivery is held due to unpaid fee, update card at post-track.info` (Spam mạo danh giao hàng)
  3. `Are you free this weekend? Let's catch up over coffee` (Ham chứa "free")
  4. `We won the team tennis match today! So excited` (Ham chứa "win")

- **Kết quả thực nghiệm:**
  - Tin 1 (Spam): BoW đoán `ham` (Sai) | Embedding đoán `spam` (Đúng).
  - Tin 2 (Spam): BoW đoán `ham` (Sai) | Embedding đoán `ham` (Sai do ngữ cảnh quá tinh vi giống thông báo bưu điện thật).
  - Tin 3 (Ham): BoW đoán `ham` (Đúng) | Embedding đoán `ham` (Đúng).
  - Tin 4 (Ham): BoW đoán `ham` (Đúng) | Embedding đoán `ham` (Đúng).

- **Giải thích:**
  - Ở tin spam né từ khoá (Tin 1), BoW hoàn toàn bất lực vì các từ "gift", "reserved", "claim-item" không đủ kích hoạt ngưỡng spam nếu thiếu các từ mạnh như "free/win". Ngược lại, Embedding nhận diện được vector ngữ nghĩa câu nằm gần cụm quà tặng lừa đảo nên phân loại chính xác.
  - Nhìn chung, mô hình Embedding thể hiện khả năng hiểu ngữ nghĩa tổng thể vượt trội hơn hẳn BoW khi đối mặt với các câu đánh lừa từ vựng.

---

## 6. Phương án cho sản phẩm thật

Với bài toán nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây), yêu cầu nghiêm ngặt **p95 < 50 ms/tin**:

1. **Phương án lựa chọn:** Chọn **3b. Encoder (MiniLM) + LogReg**.
2. **Dẫn chứng số liệu thực nghiệm:**
   - **Độ trễ:** Đạt yêu cầu SLA với $p95 = 40.33\text{ ms} < 50\text{ ms}$ (và $p50 = 28.81\text{ ms}$).
   - **Chất lượng:** Độ chính xác cao vượt trội với $F1 = 0.964$ trên tập test thường và $Recall = 0.979$ (bắt được gần 98% tin rác).
   - **Chi phí:** $0 USD tiền bản quyền API vì mô hình MiniLM hoàn toàn mã nguồn mở, tự host trên máy chủ nội bộ.
3. **Đánh đổi:** 
   - Chấp nhận thời gian chuẩn bị encode ban đầu lâu hơn BoW (~111 giây cho tập train), nhưng đổi lại chất lượng phân loại bền vững hơn hẳn trước các tin nhắn ngụy trang.
   - Không chọn LLM vì độ trễ p95 của LLM lên tới **2,119.86 ms** (vi phạm nghiêm trọng SLA 50ms) và chi phí ước tính khoảng $\$10.50 \times 10 = \$105/\text{ngày} \approx \mathbf{\$3,150/\text{tháng}}$ (quá đắt đỏ khi vận hành quy mô 10 triệu tin).

---

## 7. AI log và điều đã học

### AI Log:
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 1** | Hỏi nguyên nhân hàm `train_logreg` bị báo "chưa làm" dù đã viết code. | AI chỉ ra vẫn còn dòng lệnh `raise NotImplementedError` sót lại trong vòng lặp `for`. | Xoá bỏ dòng `raise`, chạy lại test và nhận tích xanh `✅ 1.3 train_logreg`. |
| **Phần báo cáo này nè** | kêu nó viết báo cáo dựa trên cái e đã làm | AI viết | e đọc lại để xác nhận và viết cái phần này |

### Điều hiểu ra hôm nay:
1. **Độ trễ và chi phí quan trọng không kém gì F1:** Trong hệ thống thực tế (production), một mô hình có F1 = 1.0 nhưng latency 2 giây và tốn hàng ngàn USD/tháng như LLM sẽ không thể áp dụng cho bài toán thời gian thực như lọc SMS viễn thông.
2. **Không phải mô hình càng lớn là càng tốt:** Một mô hình encoder nhỏ gọn như `all-MiniLM-L6-v2` kết hợp cùng Logistic Regression đơn giản vừa đạt SLA dưới 50ms, vừa cho F1 đạt 0.964 và chi phí bằng 0.
3. **Hiện tượng ảo tưởng chất lượng (Over-optimism):** Điểm số trên tập test thường rất dễ đánh lừa nếu dữ liệu có trùng lặp hoặc cùng phân phối. Chỉ khi stress test trên 50 case đối kháng (adversarial), điểm yếu chí mạng của từng phương pháp mới bộc lộ rõ ràng.
