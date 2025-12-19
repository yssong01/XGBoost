import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# ==========================================
# 사용자 설정
# ==========================================
# 이미지가 저장된 폴더 경로 (이전 코드에서 사용한 폴더명과 일치해야 함)
IMAGE_FOLDER = 'dashboard_GB_imgs' 
PLAY_INTERVAL = 200  # 재생 속도 (밀리초 단위, 200ms = 0.2초)

class DashboardPlayer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = self.load_image_paths()
        self.num_frames = len(self.image_files)
        
        if self.num_frames == 0:
            print(f"!!! 오류: '{folder_path}' 폴더에 이미지가 없습니다.")
            print("먼저 대시보드 생성 코드를 실행하여 이미지를 저장해주세요.")
            return

        self.current_frame = 0
        self.is_playing = False
        
        # --- Figure 및 Axis 설정 ---
        # 하단에 컨트롤러(슬라이더, 버튼) 공간을 확보하기 위해 bottom=0.2 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2) 
        
        # 초기 이미지 표시 (첫 번째 프레임)
        initial_img = mpimg.imread(self.image_files[0])
        self.im_obj = self.ax.imshow(initial_img)
        self.ax.axis('off') # 축 숨김
        self.update_title()

        # --- 위젯 설정 (Slider & Button) ---
        # 1. Slider (스크롤 바) 위치: [left, bottom, width, height]
        ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(
            ax=ax_slider,
            label='Stage ',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1,
            color='blue'
        )
        self.slider.on_changed(self.on_slider_change)

        # 2. Play/Pause 버튼 위치
        ax_button = plt.axes([0.85, 0.08, 0.1, 0.04])
        self.btn_play = Button(ax_button, 'Play', hovercolor='0.975')
        self.btn_play.on_clicked(self.toggle_play)

        # --- 애니메이션 설정 ---
        # FuncAnimation을 사용하여 주기적으로 update_animation 호출
        # 처음에는 멈춘 상태(event_source.stop)로 시작
        self.anim = animation.FuncAnimation(
            self.fig, 
            self.update_animation, 
            frames=self.num_frames, 
            interval=PLAY_INTERVAL, 
            blit=False,
            repeat=True
        )
        self.anim.event_source.stop() 

        print(f"!! 플레이어 실행됨: 총 {self.num_frames}개의 이미지를 로드했습니다.")
        plt.show()

    def load_image_paths(self):
        """폴더 내의 png 파일을 Stage 번호 순서대로 정렬하여 로드"""
        if not os.path.exists(self.folder_path):
            print(f"!!! 오류: 폴더를 찾을 수 없습니다 -> {self.folder_path}")
            return []

        files = [f for f in os.listdir(self.folder_path) if f.endswith('.png')]
        
        # 정렬: 파일명에 포함된 숫자를 기준으로 정렬 (stage_2가 stage_10보다 앞에 오도록)
        def extract_number(filename):
            match = re.search(r'stage_(\d+)', filename)
            return int(match.group(1)) if match else 0
            
        files.sort(key=extract_number)
        
        # 전체 경로로 변환
        full_paths = [os.path.join(self.folder_path, f) for f in files]
        return full_paths

    def update_image(self, frame_idx):
        """화면에 표시되는 이미지를 업데이트"""
        frame_idx = int(max(0, min(frame_idx, self.num_frames - 1)))
        self.current_frame = frame_idx
        
        # 이미지 데이터 교체
        img_path = self.image_files[frame_idx]
        img = mpimg.imread(img_path)
        self.im_obj.set_data(img)
        
        self.update_title()
        self.fig.canvas.draw_idle()

    def update_title(self):
        """현재 파일명으로 제목 업데이트"""
        filename = os.path.basename(self.image_files[self.current_frame])
        self.ax.set_title(f"Dashboard Viewer - GB {filename} ({self.current_frame + 1}/{self.num_frames})", fontsize=14, fontweight='bold')

    def on_slider_change(self, val):
        """슬라이더를 움직였을 때 호출"""
        self.update_image(val)

    def toggle_play(self, event):
        """Play/Pause 버튼 클릭 시 호출"""
        if self.is_playing:
            self.anim.event_source.stop()
            self.btn_play.label.set_text('Play')
            self.is_playing = False
        else:
            self.anim.event_source.start()
            self.btn_play.label.set_text('Pause')
            self.is_playing = True

    def update_animation(self, frame):
        """애니메이션 타이머에 의해 주기적으로 호출 (다음 프레임으로 이동)"""
        if self.is_playing:
            next_frame = (self.current_frame + 1) % self.num_frames
            # 슬라이더 값을 변경하면 on_slider_change가 트리거되어 이미지도 함께 업데이트됨
            self.slider.set_val(next_frame) 
        return self.im_obj,

if __name__ == "__main__":
    # 플레이어 실행
    DashboardPlayer(IMAGE_FOLDER)