import time

input_file = r"dev\train.csv"  # Đường dẫn file gốc 4GB
output_file = r"dev\train_2017.csv"  # Đường dẫn file mới (chỉ chứa data từ 2017)
rows_to_skip = 66458909  # Số dòng cần bỏ qua

print("Bắt đầu xử lý file. Vui lòng đợi vài phút...")
start_time = time.time()

with (
    open(input_file, "r", encoding="utf-8") as infile,
    open(output_file, "w", encoding="utf-8") as outfile,
):

    # 1. Đọc và ghi luôn dòng Header (tên cột) sang file mới
    header = infile.readline()
    outfile.write(header)

    # 2. Bỏ qua (Lazy Delete) 66,458,909 dòng tiếp theo
    print(f"Đang bỏ qua {rows_to_skip:,} dòng đầu tiên...")
    for _ in range(rows_to_skip):
        next(infile, None)  # Đọc qua nhưng không lưu vào RAM

    # 3. Ghi toàn bộ dữ liệu còn lại (từ dòng 66,458,910 trở đi) sang file mới
    print("Đang ghi dữ liệu còn lại sang file mới...")
    for line in infile:
        outfile.write(line)

end_time = time.time()
print(f"Hoàn tất! Đã tạo file {output_file} trong {end_time - start_time:.2f} giây.")
