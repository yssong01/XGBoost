import os
import sys
import re
import matplotlib
# 대용량 애니메이션 수용 (150MB까지 확장)
matplotlib.rcParams['animation.embed_limit'] = 150.0 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import matplotlib.animation as animation

# ==========================================
# [설정] 시나리오 마스터 데이터
# ==========================================
SCENARIOS = {
    1:  ('G1_Orthogonality', 'Alcohol', 'Flavanoids'),
    2:  ('G1_Orthogonality', 'Flavanoids', 'Malic Acid'),
    3:  ('G2_Redundancy', 'Flavanoids', 'Total Phenols'),
    4:  ('G2_Redundancy', 'Flavanoids', 'Od280/Od315 Of Diluted Wines'),
    5:  ('G3_Scale', 'Proline', 'Flavanoids'),
    6:  ('G3_Scale', 'Proline', 'Color Intensity'),
    7:  ('G4_Noise', 'Ash', 'Magnesium'),
    8:  ('G4_Noise', 'Color Intensity', 'Hue'),
    9:  ('G5_Physics', 'Alcohol', 'Proline'),
    10: ('G5_Physics', 'Total Phenols', 'Hue')
}

class UltimateScenarioPlayer:
    def __init__(self):
        self.scenario_map = {}
        self.scenario_names = []
        for k, v in SCENARIOS.items():
            group, x_col, y_col = v
            label = f"S{k}: {x_col}-{y_col}"
            self.scenario_names.append(label)
            # 기본 폴더명 규칙 생성
            safe_x = x_col.lower().replace(' ', '_').replace('/', '_')
            folder_base = f"{group}_Scenario{k}_{safe_x}"
            self.scenario_map[label] = folder_base

        self.current_scenario_label = self.scenario_names[0]
        self.current_model = "XGB"
        self.image_files = []
        self.image_cache = {} 
        self.current_frame = 0
        self.is_playing = False
        self.is_loop = True 
        self.anim_interval = 200 

        try:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.base_path = os.getcwd()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        # 박사님의 기존 레이아웃 수치 유지
        plt.subplots_adjust(left=0.18, bottom=0.15, right=0.95, top=0.90)

        self.load_current_scenario_files()
        img = self.get_image_data(0)
        self.im_obj = self.ax.imshow(img if img is not None else [[1]], cmap='gray')
        self.ax.axis('off')

        # --- 위젯 배치 (기존 수치 100% 동일 유지) ---
        ax_radio_s = plt.axes([0.01, 0.25, 0.10, 0.55], facecolor='#f0f0f0')
        self.radio_scenario = RadioButtons(ax_radio_s, self.scenario_names)
        self.radio_scenario.on_clicked(self.change_scenario)

        
        # [수정 전] 
        # ax_radio_m = plt.axes([0.01, 0.85, 0.10, 0.06], facecolor='#f0f0f0')
        # self.radio_model = RadioButtons(ax_radio_m, ('XGB', 'GBM'))

        # [수정 후] 4가지 모델 조합으로 확장
        # 레이아웃을 위해 높이(0.06 -> 0.12)를 살짝 늘리는 것을 추천합니다.
        ax_radio_m = plt.axes([0.01, 0.80, 0.10, 0.12], facecolor='#f0f0f0') 
        self.radio_model = RadioButtons(ax_radio_m, ('Expert_XGB', 'Expert_GBM', 'Original_XGB', 'Original_GBM'))
        self.radio_model.on_clicked(self.change_model)


        self.radio_model.on_clicked(self.change_model)

        ax_slider = plt.axes([0.35, 0.15, 0.4, 0.03])
        self.slider = Slider(ax_slider, 'Stage', 0, max(1, len(self.image_files)-1), valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        btn_y, btn_h, start_x = 0.08, 0.06, 0.35
        self.btn_prev = Button(plt.axes([start_x, btn_y, 0.04, btn_h]), '<')
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next = Button(plt.axes([start_x + 0.05, btn_y, 0.04, btn_h]), '>')
        self.btn_next.on_clicked(self.next_frame)
        self.btn_play = Button(plt.axes([start_x + 0.11, btn_y, 0.06, btn_h]), 'Play', color='honeydew')
        self.btn_play.on_clicked(self.play)
        self.btn_pause = Button(plt.axes([start_x + 0.18, btn_y, 0.06, btn_h]), 'Pause', color='oldlace')
        self.btn_pause.on_clicked(self.pause)
        self.btn_stop = Button(plt.axes([start_x + 0.25, btn_y, 0.06, btn_h]), 'Stop', color='mistyrose')
        self.btn_stop.on_clicked(self.stop)
        self.check_loop = CheckButtons(plt.axes([start_x + 0.32, btn_y, 0.07, btn_h]), ['Loop'], [True])
        self.check_loop.on_clicked(self.toggle_loop)

        # 저장 버튼들 (정렬된 수치 유지)
        self.btn_save_gif = Button(plt.axes([0.80, 0.18, 0.08, 0.05]), 'Save GIF', color='lavender')
        self.btn_save_gif.on_clicked(self.save_as_gif)
        self.btn_save_html = Button(plt.axes([0.80, 0.12, 0.08, 0.05]), 'Save HTML', color='lightyellow')
        self.btn_save_html.on_clicked(self.save_as_html)
        self.btn_auto_gif = Button(plt.axes([0.80, 0.06, 0.08, 0.05]), 'All Save GIF', color='lightcyan')
        self.btn_auto_gif.on_clicked(self.auto_save_all_gif)

        self.anim = animation.FuncAnimation(self.fig, self.update_anim, interval=self.anim_interval, cache_frame_data=False)
        self.anim.event_source.stop()
        
        self.update_title()
        plt.show()

    def load_current_scenario_files(self):
        label = self.current_scenario_label
        # [수정 1] S1, S10 등에서 숫자만 정확히 추출 (S10 -> 10)
        s_idx = re.search(r'S(\d+):', label).group(1)
        target_pattern = f"Scenario{s_idx}"
        
        try:
            category, pure_model = self.current_model.split('_')
        except ValueError:
            category, pure_model = "Expert", self.current_model

        # [수정 2] 박사님의 폴더명은 첫 글자가 대문자입니다. lower()를 제거합니다.
        # safe_x = x_col.lower().replace... -> x_col.replace...
        group, x_col, y_col = SCENARIOS[int(s_idx)]
        safe_x = x_col.replace(' ', '_').replace('/', '_')
        
        # 실제 생성된 폴더명 규칙: {Group}_Scenario{k}_{Feature}
        folder_base = f"{group}_Scenario{s_idx}_{safe_x}"
        
        # 1차 시도: 생성된 규칙대로 폴더명 조합
        # 예: G1_Orthogonality_Scenario1_Alcohol_Original_XGB
        folder_name = f"{folder_base}_{category}_{pure_model}"
        folder_path = os.path.join(self.base_path, folder_name)
        
        # 2차 시도: Fallback 검색 (유연한 매칭)
        if not os.path.exists(folder_path):
            if os.path.exists(self.base_path):
                all_dirs = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
                # 모델명(XGB/GBM)과 시나리오 번호, Original 유무를 동시에 체크
                if category == "Original":
                    matched = [d for d in all_dirs if target_pattern in d and pure_model in d and "Original" in d]
                else:
                    # Expert 폴더는 명시적으로 'Original'이 없는 폴더를 선택
                    matched = [d for d in all_dirs if target_pattern in d and pure_model in d and "Original" not in d]
                
                if matched:
                    folder_path = os.path.join(self.base_path, matched[0])

        self.image_files = []
        if os.path.exists(folder_path):
            # [수정 3] 파일 정렬 로직 강화
            # 파일명 예: ..._stage_100.png 에서 마지막 숫자(100)를 기준으로 정렬
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            if files:
                def extract_number(f):
                    nums = re.findall(r'\d+', os.path.basename(f))
                    return int(nums[-1]) if nums else 0
                
                files.sort(key=extract_number)
                self.image_files = files
                print(f"✅ 성공적으로 로드됨: {folder_path} ({len(files)} frames)")
        else:
            print(f"!! [오류] 폴더를 찾을 수 없습니다: {folder_path}")

    def get_image_data(self, idx):
        if not self.image_files or idx >= len(self.image_files): return None
        if idx in self.image_cache: return self.image_cache[idx]
        img = mpimg.imread(self.image_files[idx])
        self.image_cache[idx] = img
        return img

    def prev_frame(self, event):
        if self.current_frame > 0: self.slider.set_val(self.current_frame - 1)
    def next_frame(self, event):
        if self.current_frame < len(self.image_files) - 1: self.slider.set_val(self.current_frame + 1)
    def play(self, event):
        if not self.is_playing and self.image_files:
            self.is_playing = True; self.anim.event_source.start()
    def pause(self, event):
        self.is_playing = False; self.anim.event_source.stop()
    def stop(self, event):
        self.pause(None); self.slider.set_val(0)
    def change_scenario(self, label):
        self.current_scenario_label = label; self.refresh()
    def change_model(self, label):
        self.current_model = label; self.refresh()
    def refresh(self):
        self.pause(None); self.load_current_scenario_files(); self.image_cache = {}
        self.slider.valmax = max(1, len(self.image_files) - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax); self.slider.set_val(0); self.update_image_display(0)
    def on_slider_change(self, val):
        self.update_image_display(int(val))
    def update_image_display(self, idx):
        self.current_frame = idx
        img = self.get_image_data(idx)
        if img is not None:
            self.im_obj.set_data(img); self.update_title(); self.fig.canvas.draw_idle()
    def update_anim(self, frame):
        if not self.is_playing: return
        next_f = self.current_frame + 1
        if next_f >= len(self.image_files):
            if self.is_loop: next_f = 0
            else: self.pause(None); return
        self.slider.set_val(next_f)
    def toggle_loop(self, label): self.is_loop = not self.is_loop
    def update_title(self):
        status = f"{self.current_frame + 1}/{len(self.image_files)}" if self.image_files else "Empty"
        self.ax.set_title(f"[{self.current_model}] {self.current_scenario_label} | Frame: {status}", fontsize=14, fontweight='bold')

    def get_custom_filename(self, extension):
        # [해결] 슬래시(/)를 언더바(_)로 치환하여 파일 이름 규칙 수정
        s_label = self.current_scenario_label
        s_num_str = s_label.split(':')[0]
        folder_base = self.scenario_map.get(s_label, "")
        group_name = "_".join(folder_base.split('_')[:2])
        var_combination = s_label.split(':')[-1].strip().replace(' ', '').replace('/', '_')
        return f"{group_name}_{s_num_str}_{var_combination}_{self.current_model}.{extension}"

    def save_as_gif(self, event):
        if not self.image_files: return
        save_dir = "play_animation_XGB_GBM"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        filename = self.get_custom_filename("gif")
        save_path = os.path.join(save_dir, filename)
        
        self.pause(None)
        from PIL import Image
        try:
            print(f"🎬 GIF 생성 중: {filename}")
            imgs = [Image.open(f).copy() for f in self.image_files]
            imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=self.anim_interval, loop=0)
            print(f"   ✅ 저장 완료!")
            for i in imgs: i.close()
        except Exception as e:
            print(f"   ❌ GIF 저장 실패: {e}")

    def save_as_html(self, event):
        if not self.image_files: return
        save_dir = "play_animation_XGB_GBM"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        filename = self.get_custom_filename("html")
        save_path = os.path.join(save_dir, filename)

        self.pause(None)
        try:
            print(f"🌐 JSHTML 생성 중: {filename}")
            js_html = self.anim.to_jshtml()
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(js_html)
            print(f"   ✅ HTML 저장 완료!")
        except Exception as e:
            print(f"   ❌ HTML 저장 실패: {e}")

    def auto_save_all_gif(self, event):
        print("\n" + "="*50)
        print("!! [전체 시나리오 GIF 일괄 저장 시작]")
        print("="*50)
        
        # [수정] 아래 한 줄을 반드시 추가해야 마지막에 에러가 나지 않습니다.
        original_scenario = self.current_scenario_label
        original_model = self.current_model  # <--- 이 줄이 누락되었습니다!
        
        self.pause(None)
        
        models_to_run = ['Expert_XGB', 'Expert_GBM', 'Original_XGB', 'Original_GBM']
        total_count = len(models_to_run) * len(self.scenario_names)
        current_count = 0  # [수정] 여기서 0으로 초기화해야 합니다.

        for model in models_to_run:
            self.change_model(model)
            for scenario in self.scenario_names:
                current_count += 1
                print(f"\n[{current_count}/{total_count}] 처리 중: {model} > {scenario}")
                self.change_scenario(scenario)
                if self.image_files:
                    self.save_as_gif(None)
                else:
                    print(f"   !!! 폴더를 찾을 수 없거나 이미지가 없어 건너뜁니다.")
        
        self.change_model(original_model)
        self.change_scenario(original_scenario)
        print("\n" + "="*50)
        print("✨ 모든 시나리오 저장이 완료되었습니다!")
        print("="*50)

if __name__ == "__main__":
    UltimateScenarioPlayer()