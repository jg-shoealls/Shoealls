import qrcode
import os

def generate_report_qr():
    # 1. report.txt에서 내용 읽기
    report_path = "report.txt"
    content = ""
    
    if not os.path.exists(report_path):
        content = "보행 분석 결과: 데이터 없음."
    else:
        # 인코딩 시도 (PowerShell 리다이렉션은 보통 UTF-16)
        encodings = ['utf-16', 'utf-8', 'cp949']
        for enc in encodings:
            try:
                with open(report_path, "r", encoding=enc) as f:
                    content = f.read()
                if content: break
            except:
                continue
    
    # 핵심 내용 요약 (QR 용량 제한 방지)
    if len(content) > 600:
        content = content[:600] + "..."

    # 2. QR 코드 생성
    qr = qrcode.QRCode(
        version=None, # 자동 조절
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(content)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    
    # 3. 이미지 저장
    qr_filename = "gait_report_qr.png"
    img.save(qr_filename)
    print(f"QR 코드가 성공적으로 생성되었습니다: {qr_filename}")

if __name__ == "__main__":
    generate_report_qr()
