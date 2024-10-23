"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import cv2
import time
from options.video_test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def save_frames_and_get_fps(video_path, output_folder):
    # start time
    start_time = time.time()
    # video file name
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_folder, video_file_name)
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    print("VideoPath:", video_path)
    print(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    
    # 動画が正しく読み込まれているか確認
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 動画ファイル名から拡張子を除いた名前を取得
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        # フレームが存在しない場合、終了
        if not ret:
            break
        
        # フレームを保存
        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    # リソースを解放
    cap.release()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished extracting {frame_count} frames from {video_path}")
    return fps

def create_video_from_frames(input_folder, output_video_path, fps):
    # start time
    start_time = time.time()
    # フレームのリストを取得
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('_fake.jpg')])
    print("frames_name", frame_files[0])
    if len(frame_files) == 0:
        print("フレーム画像が見つかりませんでした。")
        return
    print("frame len:", len(frame_files))
    # 最初のフレームから動画のサイズを取得
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, layers = first_frame.shape
    print("height", height)
    print("width", width)
    
    if first_frame is None:
        print(f"Error: Could not read the first frame {frame_files[0]}")
        return

    # 動画ライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 出力形式はmp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print("fps:",fps)

    # フレームを1つずつ動画に追加
    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        # フレームの読み込みが失敗した場合をチェック
        if frame is None:
            print(f"Error: Could not read frame {frame_file}")
            continue
        
        # フレームサイズが一致しない場合はリサイズ
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
            
        out.write(frame)

    out.release()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"動画の作成が完了しました: {output_video_path}")
    return elapsed_time
    
if __name__ == '__main__':
    start_time = time.time()
    opt = TestOptions().parse()  # get test options
    print("before opt:", opt)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.num_test = 6000 # テストの回数
    input_video_path = opt.dataroot
    print("Dataroot", input_video_path)
    # video change images
    output_frames_path = './datasets/'
    video_file_name = video_file_name = os.path.splitext(os.path.basename(input_video_path))[0]
    print("Video", video_file_name)
    dataroot_path = os.path.join(output_frames_path, video_file_name)
    start_frames = time.time()
    fps = save_frames_and_get_fps(input_video_path, output_frames_path)
    end_frames = time.time()
    frames_elapsed_time = end_frames - start_frames
    # optのdataroot 更新
    setattr(opt, 'dataroot', dataroot_path)
    print("After opt:",opt)
    results_dir = opt.results_dir
    model_name = opt.name
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    
    total_cycle_loss = 0.0
    num_images = 0
    
    rgb2thermal_start_time = time.time()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        # Calculate cycle consistency loss
        model.forward()        # perform forward pass to calculate losses

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
    if num_images > 0:
        average_cycle_loss = total_cycle_loss / num_images
        print(f'Average Cycle Consistency Loss: {average_cycle_loss}')
    rgb2thermal_end_time = time.time()
    rgb2thermal_elapsed_time = rgb2thermal_end_time - rgb2thermal_start_time
    
    # images to video
    input_images = os.path.join(results_dir, model_name+'/test_latest/images/')
    output_video = os.path.join(results_dir, model_name+f'/test_latest/thermal_{video_file_name}.mp4')
    print("output_video:", output_video)
    create_elapsed_time = create_video_from_frames(input_images, output_video, fps)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"フレーム保存処理時間: {frames_elapsed_time:.2f} [s]")
    print(f"赤外線画像変換処理時間: {rgb2thermal_elapsed_time:.2f} [s]")
    print(f"動画出力処理時間: {create_elapsed_time:.2f} [s]")
    print(f"プログラム処理時間: {elapsed_time:.2f} [s]")
    # print(f"フレーム保存処理時間: {frames_elapsed_time * 1000:.2f} [ms]")
    # print(f"赤外線画像変換処理時間: {rgb2thermal_elapsed_time * 1000:.2f} [ms]")
    # print(f"動画出力処理時間: {create_elapsed_time * 1000:.2f} [ms]")
    # print(f"プログラム処理時間: {elapsed_time * 1000:.2f} [ms]")
    
    # webpage.save()  # save the HTML