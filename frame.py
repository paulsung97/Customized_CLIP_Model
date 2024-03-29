import cv2
import os

# 비디오 파일 경로와 출력 디렉토리 설정
video_file = "./videos/video.mp4"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

def extract_and_save_frames(video_path, output_directory, seconds_to_skip):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error opening video file")
        return
    
    # 초당 프레임 수(FPS)와 총 프레임 수를 구함
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 사용자가 입력한 초 단위를 프레임 단위로 변환
    skip_frames = int(fps * seconds_to_skip)
    
    frame_count = 0
    saved_frame_count = 0

    while True:
        # 현재 프레임 위치를 설정
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        ret, frame = capture.read()
        # 비디오의 끝에 도달했거나 프레임 읽기에 실패했다면 종료
        if not ret:
            break
        
        # 이미지 파일 이름 설정 및 저장
        output_path = os.path.join(output_directory, f"frame_1_{saved_frame_count}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
        
        saved_frame_count += 1
        frame_count += skip_frames

    capture.release()

# 사용자로부터 몇 초 단위로 건너뛸지 입력받음
seconds_input = float(input("Enter the number of seconds to skip between frames: "))

# 함수 호출
extract_and_save_frames(video_file, output_dir, seconds_input)
