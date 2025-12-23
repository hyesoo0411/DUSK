import os

# 입력 파일 경로
input_file = "/home/hyesoo/DUSK/data/Prof/eval/SpecificFullQA.jsonl"

# 출력 파일 경로
forget_output_file = "/home/hyesoo/DUSK/data/Prof/eval/SpecificForgetQA_listicle.jsonl"
retain_output_file = "/home/hyesoo/DUSK/data/Prof/eval/SpecificRetainQA_D2.jsonl"

# 파일 읽기 및 필터링
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 지정된 범위의 라인과 나머지를 각각 다른 파일에 저장
with open(forget_output_file, 'w', encoding='utf-8') as forget_f, \
     open(retain_output_file, 'w', encoding='utf-8') as retain_f:
    for i, line in enumerate(lines, start=1):
        # 80-100, 180-200, 280-300, ... 패턴 확인
        base = (i - 1) // 100 * 100
        if base + 80 < i <= base + 100:
            forget_f.write(line)
        else:
            retain_f.write(line)

print(f"Forget 데이터가 {forget_output_file} 에 저장되었습니다.")
print(f"Retain 데이터가 {retain_output_file} 에 저장되었습니다.")