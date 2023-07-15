import os
import subprocess
import yaml
import sys
import webbrowser
import gradio as gr
import shutil
import soundfile
import shlex

class WebUI:
    def __init__(self):
        self.train_config_path = 'configs/train.yaml'
        self.info = Info()
        self.names = []
        self.names2 = []
        self.voice_names = []
        base_config_path = 'configs/base.yaml'
        if not os.path.exists(self.train_config_path):
            shutil.copyfile(base_config_path, self.train_config_path)
            print("초기화 성공")
        else:
            print("준비됨")
        self.main_ui()

    def main_ui(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.green)) as ui:
            gr.Markdown('# so-vits-svc5.0 WebUI')

            with gr.Tab("학습"):
                with gr.Accordion('학습 안내', open=False):
                    gr.Markdown(self.info.train)
                
                gr.Markdown('### 데이터셋 파일 복사')
                with gr.Row():
                    self.dataset_name = gr.Textbox(value='', placeholder='chopin', label='데이터셋 이름', info='데이터셋 화자의 이름을 입력하세요.', interactive=True)
                    self.dataset_src = gr.Textbox(value='', placeholder='C:/Users/Tacotron2/Downloads/chopin_dataset/', label='데이터셋 폴더', info='데이터셋 wav 파일이 있는 폴더를 지정하세요.', interactive=True)
                    self.bt_dataset_copy = gr.Button(value='복사', variant="primary")

                gr.Markdown('### 전처리 파라미터 설정')
                with gr.Row():
                    self.model_name = gr.Textbox(value='sovits5.0', label='model', info='모델명', interactive=True)
                    self.f0_extractor = gr.Dropdown(choices=['crepe'], value='crepe', label='f0_extractor', info='F0 추출기', interactive=True)
                    self.thread_count = gr.Slider(minimum=1, maximum=os.cpu_count(), step=1, value=2, label='thread_count', info='전처리 스레드 수', interactive=True)

                gr.Markdown('### 학습 파라미터 설정')
                with gr.Row():
                    self.learning_rate = gr.Number(value=5e-5, label='learning_rate', info='학습률', interactive=True)
                    self.batch_size = gr.Slider(minimum=1, maximum=50, step=1, value=6, label='batch_size', info='배치 크기', interactive=True)
                    self.epochs = gr.Textbox(value='100', label='epoch', info='학습 에포크 수', interactive=True)
                with gr.Row():
                    self.info_interval = gr.Number(value=50, label='info_interval', info='학습 로깅 간격(step}', interactive=True)
                    self.eval_interval = gr.Number(value=1, label='eval_interval', info='검증 세트 간격(epoch}', interactive=True)
                    self.save_interval = gr.Number(value=5, label='save_interval', info='체크포인트 저장 간격(epoch}', interactive=True)
                    self.keep_ckpts = gr.Number(value=5, label='keep_ckpts', info='최신 체크포인트 파일 유지 갯수(0은 모두 저장)',interactive=True)
                with gr.Row():
                    self.use_pretrained = gr.Checkbox(label="use_pretrained", info='사전학습모델 사용 여부', value=True, interactive=True, visible=False)

                gr.Markdown('### 학습 시작')
                with gr.Row():
                    self.bt_open_dataset_folder = gr.Button(value='데이터 세트 폴더 열기')
                    self.bt_onekey_train = gr.Button('원클릭 학습 시작', variant="primary")
                    self.bt_tb = gr.Button('Tensorboard 열기', variant="primary")

                gr.Markdown('### 학습 재개')
                with gr.Row():
                    self.resume_model = gr.Dropdown(choices=sorted(self.names), label='Resume training progress from checkpoints', info='체크포인트에서 학습 진행 재개', interactive=True)
                    with gr.Column():
                        self.bt_refersh = gr.Button('새로 고침')
                        self.bt_resume_train = gr.Button('학습 재개', variant="primary")

            with gr.Tab("추론"):

                with gr.Accordion('추론 안내', open=False):
                    gr.Markdown(self.info.inference)

                gr.Markdown('### 추론 파라미터 설정')
                with gr.Row():
                    with gr.Column():
                        self.keychange = gr.Slider(-12, 12, value=0, step=1, label='음높이 조절')
                        self.file_list = gr.Markdown(value="", label="파일 목록")

                        with gr.Row():
                            self.resume_model2 = gr.Dropdown(choices=sorted(self.names2), label='Select the model you want to export',
                                                             info='내보낼 모델 선택', interactive=True)
                            with gr.Column():
                                self.bt_refersh2 = gr.Button(value='모델 및 사운드 새로 고침')
                                self.bt_out_model = gr.Button(value='모델 내보내기', variant="primary")
                        with gr.Row():
                            self.resume_voice = gr.Dropdown(choices=sorted(self.voice_names), label='Select the sound file',
                                                            info='*.spk.npy 파일 선택', interactive=True)
                        with gr.Row():
                            self.input_wav = gr.Audio(type='filepath', label='변환할 오디오 선택', source='upload')
                        with gr.Row():
                            self.bt_infer = gr.Button(value='변환 시작', variant="primary")
                        with gr.Row():
                            self.output_wav = gr.Audio(label='출력 오디오', interactive=False)

            self.bt_dataset_copy.click(fn=self.copydataset, inputs=[self.dataset_name, self.dataset_src])
            self.bt_open_dataset_folder.click(fn=self.openfolder)
            self.bt_onekey_train.click(fn=self.onekey_training,inputs=[self.model_name, self.thread_count,self.learning_rate,self.batch_size, self.epochs, self.info_interval, self.eval_interval,self.save_interval, self.keep_ckpts, self.use_pretrained])
            self.bt_out_model.click(fn=self.out_model, inputs=[self.model_name, self.resume_model2])
            self.bt_tb.click(fn=self.tensorboard)
            self.bt_refersh.click(fn=self.refresh_model, inputs=[self.model_name], outputs=[self.resume_model])
            self.bt_resume_train.click(fn=self.resume_train, inputs=[self.model_name, self.resume_model, self.epochs])
            self.bt_infer.click(fn=self.inference, inputs=[self.input_wav, self.resume_voice, self.keychange], outputs=[self.output_wav])
            self.bt_refersh2.click(fn=self.refresh_model_and_voice, inputs=[self.model_name],outputs=[self.resume_model2, self.resume_voice])

        ui.launch(inbrowser=True)

    def copydataset(self, dataset_name, dataset_src):
        assert dataset_name != '', '데이터셋 이름을 입력하세요'
        assert dataset_src != '', '데이터셋 경로를 입력하세요'
        assert os.path.isdir(dataset_src), '데이터셋 경로가 잘못되었습니다'
        from glob import glob
        wav_files = glob(os.path.join(dataset_src, '*.wav'))
        assert len(wav_files) > 0, '데이터셋 경로에 wav 파일이 없습니다'

        import shutil
        dst_dir = os.path.join('dataset_raw', dataset_name)
        if not os.path.exists(dst_dir): os.makedirs(dst_dir, exist_ok=True)
        for wav_file in wav_files:
            shutil.copy(wav_file, dst_dir)
        print('데이터셋 복사 완료')

    def openfolder(self):
        if not os.path.exists('dataset_raw'): os.makedirs('dataset_raw', exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile('dataset_raw')
            elif sys.platform.startswith('linux'):
                subprocess.call(['xdg-open', 'dataset_raw'])
            elif sys.platform.startswith('darwin'):
                subprocess.call(['open', 'dataset_raw'])
            else:
                print('폴더를 열지 못했습니다!')
        except BaseException:
            print('폴더를 열지 못했습니다!')

    def preprocessing(self, thread_count):
        print('전처리 시작')
        train_process = subprocess.Popen(f'{sys.executable} -u svc_preprocessing.py -t {str(thread_count)}', stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def create_config(self, model_name, learning_rate, batch_size, epochs, info_interval, eval_interval, save_interval,
                      keep_ckpts, use_pretrained):
        with open("configs/train.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['train']['model'] = model_name
        config['train']['learning_rate'] = learning_rate
        config['train']['batch_size'] = batch_size
        config['train']['epochs'] = int(epochs)
        config["log"]["info_interval"] = int(info_interval)
        config["log"]["eval_interval"] = int(eval_interval)
        config["log"]["save_interval"] = int(save_interval)
        config["log"]["keep_ckpts"] = int(keep_ckpts)
        if use_pretrained:
            config["train"]["pretrain"] = "vits_pretrain/sovits5.0.pretrain.pth"
        else:
            config["train"]["pretrain"] = ""
        with open("configs/train.yaml", "w") as f:
            yaml.dump(config, f)
        return f"로그 파라미터를 다음으로 업데이트했습니다.{config['log']}"

    def training(self, model_name):
        print('학습 시작')
        print('학습을 수행하는 새로운 콘솔 창이 열립니다.')
        print('학습 도중 학습을 중지하려면, 콘솔 창을 닫으세요.')
        train_process = subprocess.Popen(f'{sys.executable} -u svc_trainer.py -c {self.train_config_path} -n {str(model_name)}', stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def onekey_training(self, model_name, thread_count, learning_rate, batch_size, epochs, info_interval, eval_interval, save_interval, keep_ckpts, use_pretrained):
        print(model_name, thread_count, learning_rate, batch_size, epochs, info_interval, eval_interval, save_interval, keep_ckpts)
        self.create_config(model_name, learning_rate, batch_size, epochs, info_interval, eval_interval, save_interval, keep_ckpts, use_pretrained)
        self.preprocessing(thread_count)
        self.training(model_name)

    def out_model(self, model_name, resume_model2):
        print('모델 내보내기 시작')
        try:
            subprocess.Popen(f'{sys.executable} -u svc_export.py -c {self.train_config_path} -p "chkpt/{model_name}/{resume_model2}"',stdout=subprocess.PIPE)
            print('모델 내보내기 성공')
        except Exception as e:
            print("에러 발생함：", e)


    def tensorboard(self):
        tensorboard_path = os.path.join(os.path.dirname(sys.executable), 'Scripts', 'tensorboard.exe')
        tb_process = subprocess.Popen(f'{tensorboard_path} --logdir=logs --port=6006', stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        webbrowser.open("http://localhost:6006")

        while tb_process.poll() is None:
            output = tb_process.stdout.readline().decode('utf-8')
            print(output)

    def refresh_model(self, model_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, f"chkpt/{model_name}")
        self.names = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".pt"):
                    self.names.append(self.name)
            return {"choices": sorted(self.names), "__type__": "update"}
        except FileNotFoundError:
            return {"label": "모델 파일 누락", "__type__": "update"}

    def refresh_model2(self, model_name):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, f"chkpt/{model_name}")
        self.names2 = []
        try:
            for self.name in os.listdir(self.model_root):
                if self.name.endswith(".pt"):
                    self.names2.append(self.name)
            return {"choices": sorted(self.names2), "__type__": "update"}
        except FileNotFoundError as e:
            return {"label": "모델 파일 누락", "__type__": "update"}

    def refresh_voice(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_root = os.path.join(self.script_dir, "data_svc/singer")
        self.voice_names = []
        for self.name in os.listdir(self.model_root):
            if self.name.endswith(".npy"):
                self.voice_names.append(self.name)
        return {"choices": sorted(self.voice_names), "__type__": "update"}

    def refresh_model_and_voice(self, model_name):
        model_update = self.refresh_model2(model_name)
        voice_update = self.refresh_voice()
        return model_update, voice_update

    def resume_train(self, model_name, resume_model, epochs):
        print('학습 재개')
        with open("configs/train.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config['epochs'] = epochs
        with open("configs/train.yaml", "w") as f:
            yaml.dump(config, f)
        train_process = subprocess.Popen(f'{sys.executable} -u svc_trainer.py -c {self.train_config_path} -n {model_name} -p "chkpt/{model_name}/{resume_model}"', stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        while train_process.poll() is None:
            output = train_process.stdout.readline().decode('utf-8')
            print(output, end='')

    def inference(self, input, resume_voice, keychange):
        if os.path.isfile('test.wav'): os.remove('test.wav')
        self.train_config_path = 'configs/train.yaml'
        print('추론 시작')
        shutil.copy(input, ".")
        input_name = os.path.basename(input)
        os.rename(input_name, "test.wav")
        input_name = "test.wav"
        if not input_name.endswith(".wav"):
            data, samplerate = soundfile.read(input_name)
            input_name = input_name.rsplit(".", 1)[0] + ".wav"
            soundfile.write(input_name, data, samplerate)
        train_config_path = shlex.quote(self.train_config_path)
        keychange = shlex.quote(str(keychange))
        cmd = [f'{sys.executable}', "-u", "svc_inference.py", "--config", train_config_path, "--model", "sovits5.0.pth", "--spk",
               f"data_svc/singer/{resume_voice}", "--wave", "test.wav", "--shift", keychange, '--clean']
        train_process = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        print(train_process.stdout)
        print(train_process.stderr)
        print("추론 성공")
        return "svc_out.wav"


class Info:
    def __init__(self) -> None:
        self.train = '''
### 2023.7.11\n
@OOPPEENN(https://github.com/OOPPEENN)第一次编写\n
@thestmitsuk(https://github.com/thestmitsuki)二次补完\n
@OOPPEENN(https://github.com/OOPPEENN)is written for the first time\n
@thestmitsuki(https://github.com/thestmitsuki)Secondary completion

        '''
        self.inference = '''
### 2023.7.11\n
@OOPPEENN(https://github.com/OOPPEENN)第一次编写\n
@thestmitsuk(https://github.com/thestmitsuki)二次补完\n
@OOPPEENN(https://github.com/OOPPEENN)is written for the first time\n
@thestmitsuki(https://github.com/thestmitsuki)Secondary completion

        '''

def check_pretrained():
    links = {
        'hubert_pretrain/hubert-soft-0d54a1f4.pt': 'https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt',
        'speaker_pretrain/best_model.pth.tar': 'https://drive.google.com/uc?id=1UPjQ2LVSIt3o-9QMKMJcdzT8aZRZCI-E',
        'speaker_pretrain/config.json': 'https://raw.githubusercontent.com/PlayVoice/so-vits-svc-5.0/9d415f9d7c7c7a131b89ec6ff633be10739f41ed/speaker_pretrain/config.json',
        'whisper_pretrain/large-v2.pt': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt',
        'crepe/assets/full.pth': 'https://github.com/maxrmorrison/torchcrepe/raw/master/torchcrepe/assets/full.pth',
        'vits_pretrain/sovits5.0.pretrain.pth': 'https://github.com/PlayVoice/so-vits-svc-5.0/releases/download/5.0/sovits5.0.pretrain.pth',
    }

    links_to_download = {}
    for path, link in links.items():
        if not os.path.isfile(path):
            links_to_download[path] = link
    
    if len(links_to_download) == 0:
        print("사전 학습 모델이 모두 존재합니다.")
        return
    
    import gdown
    import requests

    def download(url, path):
        r = requests.get(url, allow_redirects=True)
        open(path, 'wb').write(r.content)

    for path, url in links_to_download.items():
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print(f"사전 학습 모델 {path} 다운로드 중...")
        if "drive.google.com" in url:
            gdown.download(url, path, quiet=False)
        else:
            download(url, path)
        print(f"사전 학습 모델 {path} 다운로드 완료")
    
    print("모든 사전 학습 모델이 다운로드 되었습니다.")
    return

if __name__ == "__main__":
    check_pretrained()
    webui = WebUI()
